from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid
import json

DAL_TASKS: Dict[str, Dict[str, Any]] = {}
DAL_RESULTS: Dict[str, List[Dict[str, Any]]] = {}


def coordinate_data_access(
    access_type: str, 
    data_sources: List[str], 
    agents_needed: List[str],
    requirements: str
) -> Dict[str, Any]:
    task_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    workflows = {
        "query": {
            "sequence": ["GraphQL_Query_Agent", "Data_Dashboard_Agent"],
            "description": "Query execution and visualization"
        },
        "mapping": {
            "sequence": ["Schema_Mapping_Agent", "GraphQL_Query_Agent", "Data_Dashboard_Agent"],
            "description": "Schema mapping, query, and visualization"
        },
        "performance": {
            "sequence": ["Trace_Pruning_Agent", "GraphQL_Query_Agent", "Data_Dashboard_Agent"],
            "description": "Performance analysis and reporting"
        },
        "comprehensive": {
            "sequence": ["Trace_Pruning_Agent", "Schema_Mapping_Agent", "GraphQL_Query_Agent", "Data_Dashboard_Agent"],
            "description": "Full DAL analysis workflow"
        },
        "custom": {
            "sequence": agents_needed,
            "description": f"Custom workflow with {len(agents_needed)} agents"
        }
    }
    
    workflow = workflows.get(access_type, workflows["custom"])
    
    DAL_TASKS[task_id] = {
        "task_id": task_id,
        "access_type": access_type,
        "data_sources": data_sources,
        "agents_needed": agents_needed,
        "workflow_sequence": workflow["sequence"],
        "requirements": requirements,
        "status": "initiated",
        "created_at": timestamp,
        "progress": {
            "current_step": 0,
            "total_steps": len(workflow["sequence"]),
            "completed_agents": [],
            "pending_agents": workflow["sequence"].copy()
        }
    }
    
    DAL_RESULTS[task_id] = []
    
    return {
        "status": "success",
        "task_id": task_id,
        "workflow": workflow,
        "message": f"Initiated {access_type} access with {len(workflow['sequence'])} agents"
    }


def get_access_status(task_id: Optional[str] = None) -> Dict[str, Any]:
    if task_id:
        if task_id not in DAL_TASKS:
            return {
                "status": "error",
                "message": f"Task {task_id} not found",
                "available_tasks": list(DAL_TASKS.keys())
            }
        
        task = DAL_TASKS[task_id]
        results = DAL_RESULTS.get(task_id, [])
        
        progress_percent = (task["progress"]["current_step"] / task["progress"]["total_steps"]) * 100
        
        return {
            "status": "success", 
            "task_id": task_id,
            "access_type": task["access_type"],
            "current_status": task["status"],
            "progress": {
                "percent_complete": round(progress_percent, 1),
                "current_step": task["progress"]["current_step"],
                "total_steps": task["progress"]["total_steps"],
                "completed_agents": task["progress"]["completed_agents"],
                "pending_agents": task["progress"]["pending_agents"]
            },
            "results_count": len(results),
            "created_at": task["created_at"]
        }
    else:
        all_tasks = []
        for tid, task in DAL_TASKS.items():
            progress_percent = (task["progress"]["current_step"] / task["progress"]["total_steps"]) * 100
            all_tasks.append({
                "task_id": tid,
                "access_type": task["access_type"], 
                "status": task["status"],
                "progress_percent": round(progress_percent, 1),
                "agents_involved": len(task["workflow_sequence"]),
                "created_at": task["created_at"]
            })
        
        return {
            "status": "success",
            "total_tasks": len(all_tasks),
            "active_tasks": [t for t in all_tasks if t["status"] in ["initiated", "in_progress"]],
            "completed_tasks": [t for t in all_tasks if t["status"] == "completed"],
            "all_tasks": all_tasks
        }


def generate_dal_report(
    task_id: str,
    include_raw_data: bool = False,
    report_format: str = "summary"
) -> Dict[str, Any]:
    if task_id not in DAL_TASKS:
        return {
            "status": "error",
            "message": f"Task {task_id} not found",
            "available_tasks": list(DAL_TASKS.keys())
        }
    
    task = DAL_TASKS[task_id]
    results = DAL_RESULTS.get(task_id, [])
    
    task["status"] = "generating_report"
    
    if report_format == "summary":
        report = _generate_summary_report(task, results)
    elif report_format == "detailed":
        report = _generate_detailed_report(task, results, include_raw_data)
    else:
        report = _generate_summary_report(task, results)
    
    task["status"] = "completed"
    task["completed_at"] = datetime.now().isoformat()
    
    return {
        "status": "success",
        "task_id": task_id,
        "report_format": report_format,
        "report": report,
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "duration": _calculate_duration(task),
            "agents_involved": task["workflow_sequence"],
            "data_sources_analyzed": task["data_sources"]
        }
    }


def _generate_summary_report(task: Dict[str, Any], results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "title": f"COMRADES DAL Report: {task['access_type'].title()}",
        "overview": {
            "access_type": task["access_type"],
            "scope": f"Analysis of {len(task['data_sources'])} data sources",
            "duration": _calculate_duration(task),
            "status": "Completed"
        },
        "key_findings": [
            "DAL pipeline executed successfully",
            f"Processed {len(results)} agent responses",
            "All requested components delivered"
        ],
        "recommendations": [
            "Review agent outputs for implementation guidance",
            "Monitor DAL performance regularly",
            "Implement suggested optimizations"
        ]
    }


def _generate_detailed_report(task: Dict[str, Any], results: List[Dict[str, Any]], include_raw: bool) -> Dict[str, Any]:
    report = {
        "title": f"Detailed COMRADES DAL Report: {task['access_type'].title()}",
        "task_overview": {
            "task_id": task["task_id"],
            "access_type": task["access_type"],
            "data_sources": task["data_sources"],
            "requirements": task["requirements"],
            "workflow_executed": task["workflow_sequence"]
        },
        "agent_contributions": {},
        "synthesis": {
            "data_quality": "High",
            "coverage": f"Complete - {len(task['workflow_sequence'])} agents executed",
            "insights_generated": len(results)
        }
    }
    
    for i, agent in enumerate(task["workflow_sequence"]):
        report["agent_contributions"][agent] = {
            "execution_order": i + 1,
            "specialization": _get_agent_specialization(agent),
            "status": "Completed"
        }
    
    if include_raw:
        report["raw_agent_data"] = results
    
    return report


def _get_agent_specialization(agent_name: str) -> str:
    specializations = {
        "Data_Dashboard_Agent": "Data visualization and dashboard creation",
        "GraphQL_Query_Agent": "Query execution and data retrieval",
        "Schema_Mapping_Agent": "Data schema transformation and mapping",
        "Trace_Pruning_Agent": "Performance analysis and trace optimization"
    }
    return specializations.get(agent_name, "General data analysis")


def _calculate_duration(task: Dict[str, Any]) -> str:
    if "completed_at" in task:
        start = datetime.fromisoformat(task["created_at"])
        end = datetime.fromisoformat(task["completed_at"])
        duration = end - start
        return f"{duration.total_seconds():.1f} seconds"
    return "In progress"
