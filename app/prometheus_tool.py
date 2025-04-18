import os
import httpx
import logging
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from typing import Type, Optional, List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL")
if not PROMETHEUS_URL:
    logger.error("PROMETHEUS_URL environment variable not set.")
    # Provide a default or raise an error depending on desired behavior
    PROMETHEUS_URL = "http://localhost:9090" # Example default

class PrometheusQueryInput(BaseModel):
    query: str = Field(description="The PromQL query string to execute.")
    extract_value_only: bool = Field(False, description="If true, attempt to extract only the numerical value from the result (for scalar/vector types).")

class PrometheusQueryTool(BaseTool):
    name: str = "prometheus_query"
    description: str = (
        "Useful for executing PromQL queries against a Prometheus server. "
        "Input should be a valid PromQL query string. "
        "Can optionally extract only the numerical value if the query returns a single scalar or vector result by setting 'extract_value_only' to true. "
        "Example metrics: container_cpu_usage_seconds_total, node_memory_MemAvailable_bytes, cadvisor_version_info, up{job='node-exporter'}"
    )
    args_schema: Type[BaseModel] = PrometheusQueryInput

    def _extract_single_value(self, result: List[Any], result_type: str) -> Optional[str]:
        """Attempts to extract a single numerical value from vector or scalar results."""
        value = None
        try:
            if result_type == "scalar":
                if len(result) == 2:
                    value = result[1] # Value is the second element
            elif result_type == "vector":
                if len(result) == 1:
                    # Ensure 'value' field exists and has two elements [timestamp, value]
                    if 'value' in result[0] and len(result[0]['value']) == 2:
                         value = result[0]['value'][1] # Value is the second element
                elif len(result) == 0:
                    logger.warning("Attempted to extract single value, but query returned no vector results.")
                    return "(No result)"
                else:
                    # Multiple vector results, cannot extract a single value cleanly
                    logger.warning(f"Attempted to extract single value, but query returned {len(result)} results. Returning the first value.")
                    # Fallback: return the first value if available
                    if 'value' in result[0] and len(result[0]['value']) == 2:
                        value = result[0]['value'][1]
                    else:
                        return "(Multiple results - value unclear)"

            if value is not None:
                # Try to format as float with 2 decimal places
                return f"{float(value):.2f}"
            else:
                logger.warning(f"Could not extract value from {result_type} result: {result}")
                return "(Value not found)"

        except (ValueError, TypeError, IndexError) as e:
            logger.error(f"Error formatting extracted value '{value}': {e}")
            return f"(Error formatting value: {value})"

    def _format_vector_result(self, result: List[Dict[str, Any]]) -> str:
        """Formats instant vector results."""
        if not result:
            return "No data points found for the query."

        formatted_lines = []
        for item in result:
            metric_str = ", ".join(f'{k}="{v}"' for k, v in item.get("metric", {}).items())
            timestamp, value = item.get("value", [None, "N/A"])
            if timestamp:
                dt_object = datetime.fromtimestamp(float(timestamp))
                time_str = dt_object.strftime("%Y-%m-%d %H:%M:%S")
                formatted_lines.append(f"- {{{metric_str}}} => {value} @ {time_str}")
            else:
                formatted_lines.append(f"- {{{metric_str}}} => {value}")
        return "\n".join(formatted_lines)

    def _format_matrix_result(self, result: List[Dict[str, Any]]) -> str:
        """Formats range vector results."""
        if not result:
            return "No data series found for the query."

        formatted_lines = []
        for item in result:
            metric_str = ", ".join(f'{k}="{v}"' for k, v in item.get("metric", {}).items())
            formatted_lines.append(f"Series: {{{metric_str}}}")
            values = item.get("values", [])
            if not values:
                formatted_lines.append("  No data points in this series.")
                continue
            for timestamp, value in values:
                 dt_object = datetime.fromtimestamp(float(timestamp))
                 time_str = dt_object.strftime("%Y-%m-%d %H:%M:%S")
                 formatted_lines.append(f"  - {time_str}: {value}")
            formatted_lines.append("-" * 20) # Separator for series
        return "\n".join(formatted_lines)

    def _format_scalar_result(self, result: List[Any]) -> str:
        """Formats scalar results."""
        if not result or len(result) < 2:
             return "Invalid scalar result format."
        timestamp, value = result
        dt_object = datetime.fromtimestamp(float(timestamp))
        time_str = dt_object.strftime("%Y-%m-%d %H:%M:%S")
        return f"Scalar Result: {value} @ {time_str}"

    def _run(self, query: str, extract_value_only: bool = False) -> str:
        api_endpoint = f"{PROMETHEUS_URL}/api/v1/query"
        params = {"query": query}
        logger.info(f"Executing PromQL query: {query} against {api_endpoint} (Extract value: {extract_value_only})" )

        try:
            response = httpx.get(api_endpoint, params=params, timeout=30.0)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            data = response.json()
            logger.debug(f"Prometheus API Response: {data}")

            if data.get("status") == "success":
                result_type = data["data"]["resultType"]
                result = data["data"]["result"]

                # Attempt to extract single value if requested and applicable type
                if extract_value_only and result_type in ["vector", "scalar"]:
                    single_value = self._extract_single_value(result, result_type)
                    if single_value is not None:
                         return single_value
                    else:
                         # Fallback to standard formatting if extraction failed
                         logger.warning(f"Failed to extract single value for type {result_type}, falling back to standard format.")

                # Standard formatting if not extracting value or if extraction failed
                if result_type == "vector":
                    return self._format_vector_result(result)
                elif result_type == "matrix":
                    return self._format_matrix_result(result)
                elif result_type == "scalar":
                     return self._format_scalar_result(result)
                else:
                     logger.warning(f"Unsupported result type: {result_type}")
                     return f"Received data with unsupported type '{result_type}': {result}"

            elif data.get("status") == "error":
                error_type = data.get("errorType", "Unknown Error")
                error_msg = data.get("error", "No error message provided.")
                logger.error(f"Prometheus query failed: {error_type} - {error_msg}")
                return f"Error executing Prometheus query:\nType: {error_type}\nMessage: {error_msg}\nQuery: {query}"
            else:
                 logger.error(f"Unknown Prometheus response status: {data.get('status')}")
                 return f"Received unexpected status from Prometheus: {data.get('status')}"

        except httpx.ConnectTimeout:
            logger.error(f"Connection to Prometheus ({PROMETHEUS_URL}) timed out.")
            return f"Error: Connection to Prometheus server ({PROMETHEUS_URL}) timed out."
        except httpx.RequestError as e:
            logger.error(f"HTTP request error querying Prometheus: {e}")
            return f"Error querying Prometheus API: {e}"
        except Exception as e:
            logger.exception(f"An unexpected error occurred while querying Prometheus: {e}")
            return f"An unexpected error occurred: {e}"

    async def _arun(self, query: str, extract_value_only: bool = False) -> str:
        # Basic async implementation using httpx.AsyncClient
        api_endpoint = f"{PROMETHEUS_URL}/api/v1/query"
        params = {"query": query}
        logger.info(f"(Async) Executing PromQL query: {query} against {api_endpoint} (Extract value: {extract_value_only})" )

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(api_endpoint, params=params)
                response.raise_for_status()

            data = response.json()
            logger.debug(f"(Async) Prometheus API Response: {data}")

            if data.get("status") == "success":
                result_type = data["data"]["resultType"]
                result = data["data"]["result"]

                # Attempt to extract single value if requested and applicable type
                if extract_value_only and result_type in ["vector", "scalar"]:
                    single_value = self._extract_single_value(result, result_type)
                    if single_value is not None:
                         return single_value
                    else:
                         # Fallback to standard formatting if extraction failed
                         logger.warning(f"(Async) Failed to extract single value for type {result_type}, falling back to standard format.")

                # Standard formatting if not extracting value or if extraction failed
                if result_type == "vector":
                    return self._format_vector_result(result)
                elif result_type == "matrix":
                    return self._format_matrix_result(result)
                elif result_type == "scalar":
                     return self._format_scalar_result(result)
                else:
                     logger.warning(f"(Async) Unsupported result type: {result_type}")
                     return f"Received data with unsupported type '{result_type}': {result}"

            elif data.get("status") == "error":
                error_type = data.get("errorType", "Unknown Error")
                error_msg = data.get("error", "No error message provided.")
                logger.error(f"(Async) Prometheus query failed: {error_type} - {error_msg}")
                return f"Error executing Prometheus query:\nType: {error_type}\nMessage: {error_msg}\nQuery: {query}"
            else:
                 logger.error(f"(Async) Unknown Prometheus response status: {data.get('status')}")
                 return f"Received unexpected status from Prometheus: {data.get('status')}"

        except httpx.ConnectTimeout:
            logger.error(f"(Async) Connection to Prometheus ({PROMETHEUS_URL}) timed out.")
            return f"Error: Connection to Prometheus server ({PROMETHEUS_URL}) timed out."
        except httpx.RequestError as e:
            logger.error(f"(Async) HTTP request error querying Prometheus: {e}")
            return f"Error querying Prometheus API: {e}"
        except Exception as e:
            logger.exception(f"(Async) An unexpected error occurred: {e}")
            return f"An unexpected error occurred: {e}"

# Example usage (optional, for testing)
if __name__ == "__main__":
    tool = PrometheusQueryTool()
    # Example queries:
    # print(tool.run("up{job='prometheus'}"))
    # print(tool.run("sum(rate(container_cpu_usage_seconds_total{image!=''}[1m])) by (name)"))
    # print(tool.run("node_memory_MemAvailable_bytes{instance='localhost:9100'} / 1024 / 1024")) # MB Available
    # print(tool.run("container_last_seen{name='<container_name>'}[5m]")) # Requires knowing a container name
    print(tool.run("vector(1)")) # Simple scalar test
    print(tool.run("non_existent_metric")) # Test error handling 