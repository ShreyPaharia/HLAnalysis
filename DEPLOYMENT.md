# Deployment to AWS Tokyo

Infrastructure is ready to deploy the hl-recorder to AWS ap-northeast-1 (Tokyo) for lowest latency to Hyperliquid.

## What Was Created

The following files have been added for AWS deployment:

### Infrastructure (CDK)
- `deploy/cdk/main.go` - CDK app entry point
- `deploy/cdk/stack.go` - EC2, EBS, IAM, security group definitions
- `deploy/cdk/go.mod` - Go dependencies
- `deploy/cdk/cdk.json` - CDK configuration

### Deployment Scripts
- `scripts/deploy.sh` - Deploy code updates via SSM
- `Makefile` - Convenient deployment commands

### Documentation
- `deploy/README.md` - Full deployment guide

## Quick Start

### 1. Prerequisites

Install required tools:

```bash
# AWS CLI
# macOS: brew install awscli
# Configure with your credentials
aws configure

# AWS CDK CLI
npm install -g aws-cdk

# Go 1.22+ (for CDK)
# macOS: brew install go
```

Set environment variables:

```bash
export CDK_DEFAULT_REGION=ap-northeast-1
export CDK_DEFAULT_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
```

### 2. Bootstrap CDK (one-time)

```bash
make cdk-bootstrap
```

### 3. Deploy Infrastructure

```bash
make cdk-deploy
```

This will:
- Create t4g.micro EC2 instance in Tokyo
- Attach 10GB EBS volume at `/data`
- Install Python 3.12, uv, and dependencies
- Clone this repository
- Start the recorder automatically

Wait ~5 minutes for deployment to complete.

### 4. Verify It's Running

```bash
# Check recorder status
make status

# View logs
make logs

# SSH into the instance
make ssh-ec2
```

## Cost Breakdown

- **EC2 t4g.micro**: $6.20/month (1GB RAM, ARM64)
- **EBS 10GB gp3**: $0.80/month
- **Total**: ~$7.00/month

## Architecture

```
┌─────────────────────────────────────────┐
│  AWS ap-northeast-1 (Tokyo)             │
│                                          │
│  ┌─────────────────────┐                │
│  │  EC2 t4g.micro      │                │
│  │  Amazon Linux 2023  │                │
│  │  ARM64/Graviton     │                │
│  │                     │                │
│  │  Python 3.12 + uv   │                │
│  │  systemd service    │                │
│  └─────────┬───────────┘                │
│            │                             │
│  ┌─────────▼───────────┐                │
│  │  EBS gp3 10GB       │                │
│  │  /data (Parquet)    │                │
│  └─────────────────────┘                │
│                                          │
└─────────────────────────────────────────┘
           │
           │ WebSocket
           ▼
  api.hyperliquid.xyz (~5ms)
  Binance APIs
```

## Daily Workflow

After initial deployment, to update code:

```bash
# Make changes to the code locally
git commit -am "Your changes"
git push origin main

# Deploy to EC2
make deploy
```

The deploy script will:
1. Pull latest code on EC2
2. Reinstall Python packages
3. Restart the service

## Monitoring

View recorder status and logs:

```bash
# Quick status check
make status

# Tail logs (last 100 lines)
make logs

# Interactive session
make ssh-ec2

# Once inside:
sudo systemctl status hl-recorder.service
tail -f /data/logs/recorder.log
journalctl -u hl-recorder.service -f
```

## Data Storage

Parquet files are written to `/data` with the following structure:

```
/data/
├── venue=hyperliquid/
│   ├── product_type=perp/
│   │   └── mechanism=clob/
│   │       └── event=trades/
│   │           └── symbol=BTC/
│   │               └── date=2026-05-06/
│   │                   └── hour=10/
│   │                       └── *.parquet
│   └── product_type=prediction_binary/
│       └── ...
└── venue=binance/
    └── ...
```

Check disk usage:

```bash
make ssh-ec2
df -h /data
du -sh /data/venue=*
```

### S3 Archival

Data is automatically archived to S3 (`hl-recorder-archive-<account>` in `ap-northeast-1`)
to prevent the 10 GB EBS volume from filling.

**On EC2:**

- `hl-recorder-sync.timer` runs hourly, pushes today + yesterday partitions to S3
- `hl-recorder-cleanup.timer` runs daily at 04:00 UTC: full sync, then deletes EBS
  partitions strictly older than `RETENTION_DAYS` (default `3`) that pass a
  per-partition file-count check
- Lifecycle: `Standard → Standard-IA (30d) → Deep Archive (90d)` keeps long-term
  cost near zero

Tune retention without redeploy:

```bash
make ssh-ec2
sudo systemctl edit hl-recorder-cleanup.service   # add Environment=RETENTION_DAYS=7
```

Inspect timer state and recent runs:

```bash
sudo systemctl list-timers | grep hl-recorder
sudo journalctl -u hl-recorder-sync.service -n 50
sudo journalctl -u hl-recorder-cleanup.service -n 50
```

**On your laptop:**

```bash
make pull-data        # incremental aws s3 sync s3://<bucket>/ ./data/
```

First pull is bandwidth-heavy (current data ~6 GB); subsequent pulls only
fetch new partitions. Egress to your laptop is free under the 100 GB/mo
AWS free tier.

Cost: ~$0.50/mo year 1, dropping to ~$0.10/mo steady-state once the lifecycle
moves bulk data to Deep Archive.

## Troubleshooting

### Deployment fails

Check CloudFormation:

```bash
aws cloudformation describe-stack-events \
  --stack-name HLRecorderStack \
  --max-items 20
```

### Service not starting

```bash
make ssh-ec2

# Check service status
sudo systemctl status hl-recorder.service

# View service logs
sudo journalctl -u hl-recorder.service -n 100

# Check user-data script (instance bootstrap)
sudo cat /var/log/cloud-init-output.log
```

### High latency

Verify you're in Tokyo:

```bash
make ssh-ec2
curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone
# Should show: ap-northeast-1a
```

Ping Hyperliquid:

```bash
ping -c 5 api.hyperliquid.xyz
# Should show ~5ms RTT
```

## Repository URL Configuration

The CDK stack clones from:

```
https://github.com/shreypaharia/HLAnalysis.git
```

If you fork this repo, update `repoURL` in `deploy/cdk/stack.go` line 73:

```go
repoURL := "https://github.com/YOUR_USERNAME/HLAnalysis.git"
```

Then redeploy:

```bash
make cdk-deploy
```

## Cleanup

To destroy all infrastructure:

```bash
make cdk-destroy
```

**WARNING**: This deletes the EC2 instance and creates an EBS snapshot. Download your data first if needed.

## Next Steps

1. **Deploy now**: Run `make cdk-bootstrap && make cdk-deploy`
2. **Monitor for 24h**: Check logs with `make logs`, verify data quality
3. **Tune retention if needed**: Default is `RETENTION_DAYS=3` on EBS; archived data lives in S3 indefinitely. Adjust via `sudo systemctl edit hl-recorder-cleanup.service` (see "S3 Archival" above).
4. **Pull data locally**: Run `make pull-data` to mirror the archive bucket to `./data/` for analytics
5. **Query data**: Use DuckDB to analyze recorded Parquet files

See `deploy/README.md` for more details.
