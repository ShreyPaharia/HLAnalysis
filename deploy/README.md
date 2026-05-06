# AWS Deployment (Tokyo)

Deploy the hl-recorder to AWS ap-northeast-1 (Tokyo) for lowest latency to Hyperliquid.

**Cost: ~$7.00/mo** (t4g.micro $6.20 + 10GB EBS $0.80)

## Architecture

- **EC2**: t4g.micro (1GB RAM, ARM64 Graviton)
- **Storage**: 10GB gp3 EBS mounted at `/data`
- **Access**: SSM Session Manager (no SSH keys)
- **Deployment**: Python 3.12 + systemd (no Docker)

## Prerequisites

1. AWS CLI configured with credentials
2. AWS CDK CLI installed (`npm install -g aws-cdk`)
3. Go 1.22+ installed (for CDK)
4. Set AWS region and account:
   ```bash
   export CDK_DEFAULT_REGION=ap-northeast-1
   export CDK_DEFAULT_ACCOUNT=<your-aws-account-id>
   ```

## Initial Setup (One-time)

### 1. Bootstrap CDK in Tokyo

```bash
CDK_DEFAULT_REGION=ap-northeast-1 CDK_DEFAULT_ACCOUNT=473317027471 make cdk-bootstrap
```

This provisions CDK resources in your AWS account (S3 bucket for templates, IAM roles).

### 2. Install Go dependencies

```bash
cd deploy/cdk
go mod download
cd ../..
```

### 3. Deploy infrastructure

```bash
CDK_DEFAULT_REGION=ap-northeast-1 CDK_DEFAULT_ACCOUNT=473317027471 make cdk-deploy
```

This creates:
- EC2 t4g.micro instance in Tokyo
- 10GB EBS data volume at `/data`
- Security group (outbound-only)
- IAM role with SSM access
- Installs Python, clones repo, starts recorder

The recorder starts automatically on boot.

## Daily Operations

### Deploy code changes

```bash
make deploy
```

This does:
1. Git pull latest code on EC2
2. Reinstall Python package
3. Restart systemd service

### Check recorder status

```bash
make status
```

Shows systemd service status + last 20 log lines.

### View logs

```bash
make logs
```

Shows last 100 log lines.

### SSH into EC2

```bash
make ssh-ec2
```

Opens an SSM session (no SSH key needed). Once in:

```bash
# Check service
sudo systemctl status hl-recorder.service

# View logs
tail -f /data/logs/recorder.log

# Check disk usage
df -h /data

# Query data with DuckDB
duckdb -c "SELECT venue, COUNT(*) FROM read_parquet('/data/**/*.parquet') GROUP BY venue"
```

## Infrastructure Management

### Preview changes before deploy

```bash
make cdk-diff
```

### Destroy infrastructure

```bash
make cdk-destroy
```

**WARNING**: This deletes the EC2 instance. EBS volume will be snapshotted but the snapshot costs $0.05/GB/mo. Download data first if needed.

## Updating Repository URL

If you fork this repo or use a different Git URL, update `repoURL` in `deploy/cdk/stack.go`:

```go
repoURL := "https://github.com/YOUR_USERNAME/HLAnalysis.git"
```

Then redeploy:

```bash
CDK_DEFAULT_REGION=ap-northeast-1 CDK_DEFAULT_ACCOUNT=473317027471 make cdk-deploy
```

## Optional: Enable SSH

If you need direct SSH access (SSM is recommended instead):

```bash
ENABLE_SSH=true SSH_CIDR=YOUR_IP/32 EC2_KEY_PAIR=your-key-name make cdk-deploy
```

## Troubleshooting

### Instance not starting

Check CloudFormation events:

```bash
aws cloudformation describe-stack-events --stack-name HLRecorderStack --max-items 20
```

### Service failing

SSH in and check:

```bash
make ssh-ec2

# Check service logs
sudo journalctl -u hl-recorder.service -f

# Check user-data script logs
sudo cat /var/log/cloud-init-output.log
```

### Disk full

The 10GB volume will fill in 1-2 months. Add S3 sync before then (see plan for follow-up tasks).

Check disk usage:

```bash
make ssh-ec2
df -h /data
du -sh /data/venue=*
```

## Cost Optimization

- **Spot instance**: Edit `stack.go` to use spot, saves ~70% but risks data gaps
- **Savings Plan**: 1-year commitment saves ~35%
- **S3 archival**: Sync old data to S3, save ~$3/mo after 6 months (follow-up task)

## Monitoring

Logs are written to:
- `/data/logs/recorder.log` (file, 14-day rotation)
- `journald` (systemd logs, 7-day retention)

View both:

```bash
make logs              # file logs
make ssh-ec2
sudo journalctl -u hl-recorder.service -f  # systemd logs
```
