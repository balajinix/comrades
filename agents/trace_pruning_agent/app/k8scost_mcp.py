import json
import logging
import aiohttp
from mcp.types import TextContent
from mcp.server.fastmcp import FastMCP
from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from typing import Optional, Dict, Any, List, Tuple
import os
from datetime import datetime
import threading
import uvicorn
from fastapi import FastAPI

logging.basicConfig(
    level=logging.DEBUG, 
    format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - Line:%(lineno)d - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("k8scost_mcp.log", mode='w')]
)

mcp = FastMCP("K8sCost Cloud Cost Server", host="0.0.0.0", port=8101)

# K8sCost API base URL
K8SCOST_BASE_URL = "http://localhost:9090"

# FastAPI app for serving HTML charts
chart_app = FastAPI()

# Global variable to store chart data
chart_data_store = {}

class HTMLChartGenerator:
    """Generate HTML charts using Chart.js"""
    
    @staticmethod
    def generate_bar_chart_html(data: List[Tuple[str, float]], title: str = "Cost Analysis", chart_type: str = "bar") -> str:
        """Generate HTML with Chart.js bar chart"""
        
        labels = [item[0] for item in data]
        values = [item[1] for item in data]
        
        # Generate colors for bars
        colors = [
            f'rgba({(i * 50) % 255}, {(i * 80 + 100) % 255}, {(i * 120 + 150) % 255}, 0.7)'
            for i in range(len(data))
        ]
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #2c3e50;
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            color: #7f8c8d;
            margin: 10px 0;
            font-size: 1.1em;
        }}
        .chart-container {{
            position: relative;
            height: 500px;
            margin: 20px 0;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }}
        .stat-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border-left: 4px solid #3498db;
        }}
        .stat-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .stat-label {{
            color: #7f8c8d;
            margin-top: 5px;
        }}
        .refresh-btn {{
            background: #3498db;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1em;
            margin: 20px auto;
            display: block;
            transition: background 0.3s;
        }}
        .refresh-btn:hover {{
            background: #2980b9;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 {title}</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <button class="refresh-btn" onclick="location.reload()">🔄 Refresh Data</button>
        </div>
        
        <div class="chart-container">
            <canvas id="costChart"></canvas>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">${sum(values):.2f}</div>
                <div class="stat-label">Total Cost</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${max(values) if values else 0:.2f}</div>
                <div class="stat-label">Highest Cost</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${sum(values)/len(values) if values else 0:.2f}</div>
                <div class="stat-label">Average Cost</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(values)}</div>
                <div class="stat-label">Total Items</div>
            </div>
        </div>
    </div>

    <script>
        const ctx = document.getElementById('costChart').getContext('2d');
        const chart = new Chart(ctx, {{
            type: '{chart_type}',
            data: {{
                labels: {json.dumps(labels)},
                datasets: [{{
                    label: 'Cost ($)',
                    data: {json.dumps(values)},
                    backgroundColor: {json.dumps(colors)},
                    borderColor: {json.dumps([color.replace('0.7', '1') for color in colors])},
                    borderWidth: 2,
                    borderRadius: 8,
                    borderSkipped: false,
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        display: false
                    }},
                    tooltip: {{
                        backgroundColor: 'rgba(0,0,0,0.8)',
                        titleColor: 'white',
                        bodyColor: 'white',
                        borderColor: '#3498db',
                        borderWidth: 1,
                        callbacks: {{
                            label: function(context) {{
                                return context.parsed.y.toFixed(4) + ' $';
                            }}
                        }}
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        grid: {{
                            color: 'rgba(0,0,0,0.1)'
                        }},
                        ticks: {{
                            callback: function(value) {{
                                return '$' + value.toFixed(2);
                            }}
                        }}
                    }},
                    x: {{
                        grid: {{
                            display: false
                        }},
                        ticks: {{
                            maxRotation: 45,
                            minRotation: 0
                        }}
                    }}
                }},
                animation: {{
                    duration: 1000,
                    easing: 'easeInOutQuart'
                }}
            }}
        }});
    </script>
</body>
</html>
        """
        return html_template

# Chart server routes
@chart_app.get("/", response_class=HTMLResponse)
async def show_latest_chart():
    """Show the latest generated chart"""
    if not chart_data_store:
        return HTMLResponse("""
        <html><body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
            <h2>No chart data available</h2>
            <p>Run a cost analysis tool to generate charts</p>
            <button onclick="location.reload()" style="padding: 10px 20px; background: #3498db; color: white; border: none; border-radius: 5px; cursor: pointer;">
                Refresh
            </button>
        </body></html>
        """)
    
    latest_key = max(chart_data_store.keys())
    chart_info = chart_data_store[latest_key]
    return HTMLResponse(chart_info['html'])

@chart_app.get("/chart/{chart_id}", response_class=HTMLResponse)
async def show_chart(chart_id: str):
    """Show specific chart by ID"""
    if chart_id not in chart_data_store:
        return HTMLResponse("<html><body><h2>Chart not found</h2></body></html>")
    
    chart_info = chart_data_store[chart_id]
    return HTMLResponse(chart_info['html'])

@chart_app.get("/list")
async def list_charts():
    """List all available charts"""
    charts = []
    for chart_id, info in chart_data_store.items():
        charts.append({
            "id": chart_id,
            "title": info['title'],
            "timestamp": info['timestamp'],
            "url": f"http://localhost:9000/chart/{chart_id}"
        })
    return {"charts": charts}

async def make_k8scost_request(endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Make HTTP request to K8sCost API"""
    url = f"{K8SCOST_BASE_URL}{endpoint}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise HTTPException(
                        status_code=response.status, 
                        detail=f"K8sCost API error: {response.status} - {await response.text()}"
                    )
    except aiohttp.ClientError as e:
        raise HTTPException(status_code=500, detail=f"Connection error to K8sCost: {str(e)}")

def generate_chart_and_store(data: List[Tuple[str, float]], title: str, chart_type: str = "bar") -> str:
    """Generate chart HTML and store it, return the URL"""
    chart_id = f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    html_content = HTMLChartGenerator.generate_bar_chart_html(data, title, chart_type)
    
    chart_data_store[chart_id] = {
        'html': html_content,
        'title': title,
        'timestamp': datetime.now().isoformat()
    }
    
    return f"http://localhost:9000/chart/{chart_id}"

@mcp.tool()
async def get_cloud_cost_data(window: str = "2d", aggregate: Optional[str] = None, accumulate: Optional[bool] = None, filter_services: Optional[str] = None, generate_chart: bool = True):
    """Get cloud cost data from K8sCost with optional chart generation.
    
    Args:
        window: Time window for cloud cost data (e.g., '2d', '7d', '1h', '30d'). Default is '2d'
        aggregate: Aggregation level (e.g., 'service', 'account', 'provider', 'invoiceEntityID')
        accumulate: Whether to accumulate costs over the window period
        filter_services: Filter by specific services (e.g., 'AmazonEC2', 'AmazonS3')
        generate_chart: Whether to generate HTML chart (default: True)
    
    Returns:
        Cloud cost data with optional chart URL
    """
    try:
        logging.info(f"Getting cloud cost data for window: {window}")
        
        params = {"window": window}
        if aggregate:
            params["aggregate"] = aggregate
        if accumulate is not None:
            params["accumulate"] = str(accumulate).lower()
        if filter_services:
            params["filter"] = f"service:{filter_services}"
        
        data = await make_k8scost_request("/model/cloudCost", params)
        
        has_cost_data = False
        chart_data = []
        
        if data.get("data", {}).get("sets"):
            for cost_set in data["data"]["sets"]:
                if cost_set.get("cloudCosts") is not None:
                    has_cost_data = True
                    # Extract data for charting
                    for cost_entry in cost_set.get("cloudCosts", {}):
                        for service_name, cost_details in cost_entry.items():
                            if isinstance(cost_details, dict) and "cost" in cost_details:
                                chart_data.append((service_name, cost_details["cost"]))
        
        result = {
            "status": "success",
            "data": data
        }
        
        if not has_cost_data:
            result["message"] = "API call successful, but no cloud cost data available for the specified time period"
            result["note"] = "This could be because: 1) No cloud resources were used during this period, 2) Billing data hasn't been imported yet, 3) K8sCost hasn't been configured with cloud billing data"
        else:
            if generate_chart and chart_data:
                chart_url = generate_chart_and_store(
                    chart_data, 
                    f"Cloud Cost Analysis - {window}",
                    "bar"
                )
                result["chart_url"] = chart_url
                result["message"] = f"Chart generated and available at: {chart_url}"
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
    except Exception as e:
        logging.error(f"Error getting cloud cost data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching cloud cost data: {str(e)}")

@mcp.tool()
async def get_allocation_data(
    window: str = "15m", 
    aggregate: Optional[str] = "namespace", 
    accumulate: Optional[bool] = None, 
    filter_by: Optional[str] = None,
    generate_chart: bool = True
):
    """Get allocation data from K8sCost with chart generation and cost alerts."""
    
    try:
        logging.info(f"Getting allocation data for window: {window}")
        
        params = {"window": window}
        if aggregate:
            params["aggregate"] = aggregate
        if accumulate is not None:
            params["accumulate"] = str(accumulate).lower()
        if filter_by:
            params["filter"] = filter_by

        data = await make_k8scost_request("/model/allocation", params)
        
        simplified_costs = []
        alerts = []
        cost_threshold = 50
        chart_data = []
        
        monitored_namespaces = {"opensearch", "opentelemetry-collectors", "opentelemetry-operator", "prometheus", "clickhouse"}
        monitored_costs_sum = 0

        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            for entry in data["data"]:
                if entry and isinstance(entry, dict):
                    for namespace, details in entry.items():
                        if isinstance(details, dict):
                            cpu_cost = details.get("cpuCost", 0)
                            ram_cost = details.get("ramCost", 0)
                            total_cost = cpu_cost + ram_cost

                            simplified_costs.append({
                                "namespace": namespace,
                                "cpuCost": cpu_cost,
                                "ramCost": ram_cost,
                                "totalCost": total_cost
                            })
                            
                            chart_data.append((namespace, total_cost))
                            
                            if namespace in monitored_namespaces:
                                monitored_costs_sum += total_cost

        if monitored_costs_sum > cost_threshold:
            monitored_namespaces_list = ", ".join(monitored_namespaces)
            alerts.append(
                f"Combined cost of observability namespaces ({monitored_namespaces_list}) is {monitored_costs_sum:.4f}, which exceeds the threshold of {cost_threshold}."
            )

        result = {
            "status": "success",
            "costs": simplified_costs,
            "monitored_namespaces_total": monitored_costs_sum
        }
        
        if alerts:
            result["alerts"] = alerts
        
        if generate_chart and chart_data:
            # Sort chart data by cost for better visualization
            chart_data.sort(key=lambda x: x[1], reverse=True)
            chart_url = generate_chart_and_store(
                chart_data, 
                f"Namespace Cost Allocation - {window}",
                "bar"
            )
            result["chart_url"] = chart_url
            result["message"] = f"Chart generated and available at: {chart_url}"

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logging.error(f"Error getting allocation data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching allocation data: {str(e)}")

def start_chart_server():
    """Start the chart server in a separate thread"""
    uvicorn.run(chart_app, host="0.0.0.0", port=9000, log_level="info")

def start_server():
    # Start chart server in background thread
    chart_thread = threading.Thread(target=start_chart_server, daemon=True)
    chart_thread.start()
    
    logging.info("Chart server started at http://localhost:9000")
    
    # Start MCP server
    mcp.run(transport="sse")

if __name__ == "__main__":
    start_server()
