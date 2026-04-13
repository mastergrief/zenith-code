"""
CALM AWS services knowledge backend — service names, purposes, pricing models.

Models confuse AWS services, hallucinate features, mix up service categories.
"""

from __future__ import annotations

_DATA_VERSION = "2025-01"

_SERVICES = {
    "EC2": {"name": "Elastic Compute Cloud", "category": "compute", "purpose": "Virtual servers", "pricing": ["on-demand", "reserved", "spot"], "use_when": "custom compute, full OS control"},
    "Lambda": {"name": "Lambda", "category": "compute", "purpose": "Serverless functions", "pricing": ["per-request + duration"], "use_when": "event-driven, < 15 min, no server management", "limits": "15 min timeout, 10 GB memory, 250 MB deploy package"},
    "ECS": {"name": "Elastic Container Service", "category": "compute", "purpose": "Container orchestration (Docker)", "pricing": ["per EC2/Fargate resource"], "use_when": "containerized apps, AWS-native"},
    "EKS": {"name": "Elastic Kubernetes Service", "category": "compute", "purpose": "Managed Kubernetes", "pricing": ["$0.10/hr cluster + compute"], "use_when": "K8s ecosystem, multi-cloud portability"},
    "Fargate": {"name": "Fargate", "category": "compute", "purpose": "Serverless containers (no EC2 management)", "pricing": ["per vCPU + memory per second"], "use_when": "containers without managing servers"},
    "S3": {"name": "Simple Storage Service", "category": "storage", "purpose": "Object storage (unlimited)", "pricing": ["per GB stored + requests"], "tiers": ["Standard", "IA", "Glacier", "Deep Archive"], "use_when": "files, backups, static hosting, data lake"},
    "EBS": {"name": "Elastic Block Store", "category": "storage", "purpose": "Block storage for EC2", "pricing": ["per GB/month provisioned"], "use_when": "EC2 disk, databases", "types": ["gp3", "io2", "st1", "sc1"]},
    "RDS": {"name": "Relational Database Service", "category": "database", "purpose": "Managed SQL databases", "engines": ["PostgreSQL", "MySQL", "MariaDB", "Oracle", "SQL Server", "Aurora"], "use_when": "relational data, managed backups/patching"},
    "DynamoDB": {"name": "DynamoDB", "category": "database", "purpose": "Managed NoSQL (key-value + document)", "pricing": ["per read/write unit or on-demand"], "use_when": "high-scale, low-latency, schemaless", "gotcha": "design around access patterns, not data model"},
    "Aurora": {"name": "Aurora", "category": "database", "purpose": "MySQL/PostgreSQL-compatible, AWS-optimized", "pricing": ["per ACU or provisioned"], "use_when": "high-perf relational, auto-scaling reads"},
    "ElastiCache": {"name": "ElastiCache", "category": "database", "purpose": "Managed Redis/Memcached", "use_when": "caching, sessions, pub/sub"},
    "VPC": {"name": "Virtual Private Cloud", "category": "networking", "purpose": "Isolated network in AWS", "components": ["subnets", "route tables", "NAT gateway", "internet gateway", "security groups", "NACLs"]},
    "CloudFront": {"name": "CloudFront", "category": "networking", "purpose": "CDN (content delivery)", "use_when": "static assets, API acceleration, DDoS protection"},
    "Route53": {"name": "Route 53", "category": "networking", "purpose": "DNS + domain registration", "routing": ["simple", "weighted", "latency", "failover", "geolocation"]},
    "ALB": {"name": "Application Load Balancer", "category": "networking", "purpose": "HTTP/HTTPS load balancing (layer 7)", "vs_NLB": "ALB = HTTP routing, NLB = TCP/UDP, ultra-low latency"},
    "IAM": {"name": "Identity and Access Management", "category": "security", "purpose": "Users, roles, policies (who can do what)", "free": True, "critical": True},
    "SQS": {"name": "Simple Queue Service", "category": "messaging", "purpose": "Managed message queue", "types": ["Standard (at-least-once)", "FIFO (exactly-once)"], "use_when": "decouple services, async processing"},
    "SNS": {"name": "Simple Notification Service", "category": "messaging", "purpose": "Pub/sub messaging + notifications", "use_when": "fan-out, push notifications, email/SMS alerts"},
    "CloudWatch": {"name": "CloudWatch", "category": "monitoring", "purpose": "Metrics, logs, alarms", "use_when": "monitoring, alerting, log aggregation"},
    "CloudFormation": {"name": "CloudFormation", "category": "devops", "purpose": "Infrastructure as Code (YAML/JSON)", "vs_terraform": "CloudFormation = AWS-only, Terraform = multi-cloud"},
    "CodePipeline": {"name": "CodePipeline", "category": "devops", "purpose": "CI/CD pipeline orchestration"},
    "Cognito": {"name": "Cognito", "category": "security", "purpose": "User authentication and authorization", "features": ["user pools", "identity pools", "social login", "MFA"]},
    "Step Functions": {"name": "Step Functions", "category": "compute", "purpose": "Serverless workflow orchestration", "use_when": "complex multi-step workflows, state machines"},
    "EventBridge": {"name": "EventBridge", "category": "messaging", "purpose": "Serverless event bus", "use_when": "event-driven architecture, SaaS integrations"},
}


def aws_service(name: str) -> dict:
    """Get details about an AWS service."""
    key = str(name).strip()
    entry = _SERVICES.get(key)
    if not entry:
        # Case-insensitive + partial match
        for k, v in _SERVICES.items():
            if k.lower() == key.lower() or key.lower() in v.get("name", "").lower():
                return {"service": k, **v}
        return {"error": f"Unknown: {name}", "valid": sorted(_SERVICES.keys())}
    return {"service": key, **entry}


def aws_compare(svc1: str, svc2: str) -> dict:
    """Compare two AWS services."""
    s1 = aws_service(svc1)
    s2 = aws_service(svc2)
    return {"service_1": s1, "service_2": s2}


def aws_by_category(category: str) -> list[str]:
    """List AWS services in a category."""
    cat = str(category).lower().strip()
    return sorted(k for k, v in _SERVICES.items() if v.get("category", "").lower() == cat)


def lambda_vs_ec2() -> dict:
    """Compare Lambda vs EC2."""
    return {
        "Lambda": {"management": "none", "scaling": "automatic", "pricing": "per-request", "max_duration": "15 min", "best_for": "event-driven, API backends, short tasks"},
        "EC2": {"management": "you manage OS/patching", "scaling": "manual or auto-scaling groups", "pricing": "per-hour/second", "max_duration": "unlimited", "best_for": "long-running, custom runtime, GPU"},
    }


AWS_FUNCTIONS = {
    "aws_service": aws_service,
    "aws_compare": aws_compare,
    "aws_by_category": aws_by_category,
    "lambda_vs_ec2": lambda_vs_ec2,
}

AWS_NL_PATTERNS = [
    (r'(?:what is|explain)\s+(?:AWS\s+)?(EC2|Lambda|S3|RDS|DynamoDB|ECS|EKS|Fargate|CloudFront|Route53|IAM|SQS|SNS|VPC|Aurora|CloudWatch|Cognito|Step Functions|EventBridge)', 'aws_service("{0}")'),
    (r'(?:difference between|compare|vs)\s+(?:AWS\s+)?Lambda\s+(?:and|vs)\s+EC2', 'lambda_vs_ec2()'),
    (r'(?:AWS|aws)\s+services?\s+(?:for|in)\s+(\w+)', 'aws_by_category("{0}")'),
]
