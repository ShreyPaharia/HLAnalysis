# Quick Start - Deploy to Tokyo in 5 Minutes

## Prerequisites

```bash
# Install AWS CDK CLI
npm install -g aws-cdk

# Configure AWS credentials
aws configure
# Set region: ap-northeast-1 (Tokyo)
# Enter your Access Key ID and Secret Access Key

# Export account info
export CDK_DEFAULT_REGION=ap-northeast-1
export CDK_DEFAULT_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
```

## Deploy

```bash
# 1. Bootstrap CDK (one-time, takes ~2 min)
make cdk-bootstrap

# 2. Deploy infrastructure (takes ~5 min)
make cdk-deploy
```

CDK will create:
- EC2 t4g.micro in Tokyo (ap-northeast-1)
- 10GB EBS data volume
- Python 3.12 + uv environment
- Clone this repo and start recording

## Verify

```bash
# Check status
make status

# View logs
make logs

# SSH into instance
make ssh-ec2
```

## Update Code Later

```bash
# After making changes and pushing to GitHub
make deploy
```

## Cost

- **$7/month** (EC2 $6.20 + EBS $0.80)
- Lowest latency to Hyperliquid (~5ms vs ~200ms from US)

## Next Steps

See [`DEPLOYMENT.md`](../DEPLOYMENT.md) for full documentation.
