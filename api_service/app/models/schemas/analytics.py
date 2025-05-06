"""
Schemas for analytics and business intelligence endpoints.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime, date

from app.models.schemas.common import FilterParams


class TimeSeriesPoint(BaseModel):
    """Schema for a single time series data point."""
    timestamp: str  # ISO format date or datetime
    value: float


class TimeSeriesData(BaseModel):
    """Schema for time series data."""
    label: str
    data: List[TimeSeriesPoint]
    color: Optional[str] = None  # Hex color code


class ComplianceMetric(BaseModel):
    """Schema for a single compliance metric."""
    name: str
    value: float
    unit: str
    change: Optional[float] = None  # Percentage change from previous period
    target: Optional[float] = None
    status: str  # 'good', 'warning', 'critical'


class KPIDashboard(BaseModel):
    """Schema for KPI dashboard metrics."""
    compliance_rate: ComplianceMetric
    violation_count: ComplianceMetric
    resolved_violations: ComplianceMetric
    avg_resolution_time: ComplianceMetric
    most_common_violation: str
    most_compliant_area: str
    least_compliant_area: str
    time_period: str
    
    class Config:
        schema_extra = {
            "example": {
                "compliance_rate": {
                    "name": "Overall Compliance Rate",
                    "value": 92.5,
                    "unit": "%",
                    "change": 3.2,
                    "target": 95.0,
                    "status": "good"
                },
                "violation_count": {
                    "name": "Total Violations",
                    "value": 78,
                    "unit": "count",
                    "change": -15.3,
                    "target": 50,
                    "status": "warning"
                },
                "resolved_violations": {
                    "name": "Resolved Violations",
                    "value": 65,
                    "unit": "count",
                    "change": 10.2,
                    "target": 75,
                    "status": "good"
                },
                "avg_resolution_time": {
                    "name": "Avg. Resolution Time",
                    "value": 4.2,
                    "unit": "hours",
                    "change": -0.8,
                    "target": 3.0,
                    "status": "warning"
                },
                "most_common_violation": "no_helmet",
                "most_compliant_area": "Building B",
                "least_compliant_area": "Loading Dock",
                "time_period": "Last 30 days"
            }
        }


class AnalyticsFilterParams(FilterParams):
    """Filter parameters for analytics."""
    area_id: Optional[str] = None
    time_granularity: Optional[str] = "day"  # hour, day, week, month
    group_by: Optional[str] = None  # camera, area, violation_type


class DashboardChartData(BaseModel):
    """Schema for dashboard chart data."""
    title: str
    description: Optional[str] = None
    type: str  # line, bar, pie, etc.
    series: List[TimeSeriesData]
    x_axis_label: Optional[str] = None
    y_axis_label: Optional[str] = None
    start_date: datetime
    end_date: datetime
    
    class Config:
        schema_extra = {
            "example": {
                "title": "Compliance Rate Over Time",
                "description": "Daily compliance rate for the past 30 days",
                "type": "line",
                "series": [
                    {
                        "label": "Overall Compliance",
                        "data": [
                            {"timestamp": "2025-05-01", "value": 91.2},
                            {"timestamp": "2025-05-02", "value": 92.5}
                        ],
                        "color": "#4CAF50"
                    },
                    {
                        "label": "Target",
                        "data": [
                            {"timestamp": "2025-05-01", "value": 95.0},
                            {"timestamp": "2025-05-02", "value": 95.0}
                        ],
                        "color": "#FFC107"
                    }
                ],
                "x_axis_label": "Date",
                "y_axis_label": "Compliance Rate (%)",
                "start_date": "2025-05-01T00:00:00Z",
                "end_date": "2025-05-30T23:59:59Z"
            }
        }


class ComparativeAnalysis(BaseModel):
    """Schema for comparative analysis between different areas/cameras."""
    dimension: str  # What we're comparing (cameras, areas)
    metrics: List[str]  # What metrics we're showing
    data: List[Dict[str, Any]]
    time_period: str
    
    class Config:
        schema_extra = {
            "example": {
                "dimension": "area",
                "metrics": ["compliance_rate", "violation_count", "resolved_rate"],
                "data": [
                    {
                        "name": "Building A",
                        "compliance_rate": 94.2,
                        "violation_count": 32,
                        "resolved_rate": 87.5
                    },
                    {
                        "name": "Building B",
                        "compliance_rate": 96.8,
                        "violation_count": 18,
                        "resolved_rate": 94.4
                    }
                ],
                "time_period": "Last 30 days"
            }
        }


class AnalyticsExportParams(BaseModel):
    """Parameters for exporting analytics data."""
    report_type: str  # compliance, violations, trends
    format: str  # csv, json
    start_date: date
    end_date: date
    include_details: bool = False
    group_by: Optional[str] = None  # camera, area, day, week, month
    
    class Config:
        schema_extra = {
            "example": {
                "report_type": "compliance",
                "format": "csv",
                "start_date": "2025-05-01",
                "end_date": "2025-05-30",
                "include_details": True,
                "group_by": "area"
            }
        }
