package main

import (
	"fmt"

	"github.com/aws/aws-cdk-go/awscdk/v2"
	"github.com/aws/aws-cdk-go/awscdk/v2/awsec2"
	"github.com/aws/aws-cdk-go/awscdk/v2/awsiam"
	"github.com/aws/constructs-go/constructs/v10"
	"github.com/aws/jsii-runtime-go"
)

type HLRecorderStackProps struct {
	awscdk.StackProps
	EnableSSH bool
	KeyPair   string
	SSHCIDR   string
}

func NewHLRecorderStack(scope constructs.Construct, id string, props *HLRecorderStackProps) awscdk.Stack {
	stack := awscdk.NewStack(scope, &id, &props.StackProps)

	// Default VPC (free - no new VPC created)
	vpc := awsec2.Vpc_FromLookup(stack, jsii.String("DefaultVPC"), &awsec2.VpcLookupOptions{
		IsDefault: jsii.Bool(true),
	})

	// Security Group - outbound only (no inbound except optional SSH)
	sg := awsec2.NewSecurityGroup(stack, jsii.String("HLRecorderSG"), &awsec2.SecurityGroupProps{
		Vpc:               vpc,
		SecurityGroupName: jsii.String("hl-recorder-sg"),
		Description:       jsii.String("HLAnalysis recorder - outbound only (Hyperliquid + Binance APIs)"),
		AllowAllOutbound:  jsii.Bool(true),
	})

	// Optional SSH access
	if props.EnableSSH {
		sg.AddIngressRule(
			awsec2.Peer_Ipv4(jsii.String(props.SSHCIDR)),
			awsec2.Port_Tcp(jsii.Number(22)),
			jsii.String("SSH access"),
			jsii.Bool(false),
		)
	}

	// IAM Role for EC2
	role := awsiam.NewRole(stack, jsii.String("HLRecorderRole"), &awsiam.RoleProps{
		RoleName:  jsii.String("hl-recorder-ec2-role"),
		AssumedBy: awsiam.NewServicePrincipal(jsii.String("ec2.amazonaws.com"), nil),
		ManagedPolicies: &[]awsiam.IManagedPolicy{
			awsiam.ManagedPolicy_FromAwsManagedPolicyName(jsii.String("AmazonSSMManagedInstanceCore")),
		},
	})

	// Allow EC2 to read GitHub deploy key from SSM Parameter Store
	role.AddToPolicy(awsiam.NewPolicyStatement(&awsiam.PolicyStatementProps{
		Effect:  awsiam.Effect_ALLOW,
		Actions: jsii.Strings("ssm:GetParameter"),
		Resources: jsii.Strings(
			fmt.Sprintf("arn:aws:ssm:%s:%s:parameter/hl-recorder/*", *props.Env.Region, *props.Env.Account),
		),
	}))

	// KMS decrypt for SecureString parameters
	role.AddToPolicy(awsiam.NewPolicyStatement(&awsiam.PolicyStatementProps{
		Effect:  awsiam.Effect_ALLOW,
		Actions: jsii.Strings("kms:Decrypt"),
		Resources: jsii.Strings(
			fmt.Sprintf("arn:aws:kms:%s:%s:alias/aws/ssm", *props.Env.Region, *props.Env.Account),
		),
	}))

	// User data script
	userData := awsec2.UserData_ForLinux(&awsec2.LinuxUserDataOptions{
		Shebang: jsii.String("#!/bin/bash -xe"),
	})

	repoURL := "git@github.com:ShreyPaharia/HLAnalysis.git"

	userData.AddCommands(
		// Wait for secondary volume to attach
		jsii.String("sleep 15"),

		// Identify the secondary EBS volume (the unmounted disk)
		jsii.String("ROOT_DEV=$(findmnt -n -o SOURCE /)"),
		jsii.String("ROOT_DISK=$(lsblk -no PKNAME \"$ROOT_DEV\")"),
		jsii.String("DATA_DEVICE=$(lsblk -dpn -o NAME,TYPE,MOUNTPOINT | awk -v root=\"$ROOT_DISK\" '$2==\"disk\" && $3==\"\" && $1 !~ root {print $1; exit}')"),
		jsii.String("echo \"Root disk: /dev/$ROOT_DISK, Data device: $DATA_DEVICE\""),

		// Format and mount at /data (only if not already formatted)
		jsii.String("if ! blkid \"$DATA_DEVICE\"; then mkfs -t ext4 \"$DATA_DEVICE\"; fi"),
		jsii.String("mkdir -p /data"),
		jsii.String("mountpoint -q /data || mount \"$DATA_DEVICE\" /data"),

		// Add to fstab for persistence across reboots
		jsii.String("DATA_UUID=$(blkid -s UUID -o value \"$DATA_DEVICE\")"),
		jsii.String("grep -q \"$DATA_UUID\" /etc/fstab || echo \"UUID=$DATA_UUID /data ext4 defaults,nofail 0 2\" >> /etc/fstab"),

		// Create logs directory
		jsii.String("mkdir -p /data/logs"),
		jsii.String("chown -R ec2-user:ec2-user /data"),

		// Install Python 3.12, git, and development tools
		jsii.String("dnf install -y python3.12 python3.12-pip git gcc python3.12-devel"),

		// Install uv (fast Python package installer; installs to /root/.local/bin)
		jsii.String("curl -LsSf https://astral.sh/uv/install.sh | sh"),

		// Set up GitHub deploy key from SSM Parameter Store
		jsii.String("mkdir -p /home/ec2-user/.ssh"),
		jsii.String(fmt.Sprintf("aws ssm get-parameter --name /hl-recorder/github-deploy-key --with-decryption --region %s --query Parameter.Value --output text > /home/ec2-user/.ssh/github_deploy_key", *props.Env.Region)),
		jsii.String("chmod 600 /home/ec2-user/.ssh/github_deploy_key"),
		jsii.String(`cat > /home/ec2-user/.ssh/config << SSHEOF
Host github.com
  HostName github.com
  User git
  IdentityFile /home/ec2-user/.ssh/github_deploy_key
  IdentitiesOnly yes
  StrictHostKeyChecking accept-new
SSHEOF`),
		jsii.String("chmod 600 /home/ec2-user/.ssh/config"),
		jsii.String("chown -R ec2-user:ec2-user /home/ec2-user/.ssh"),

		// Clone the repository as ec2-user
		jsii.String("mkdir -p /opt/hl-recorder"),
		jsii.String("chown ec2-user:ec2-user /opt/hl-recorder"),
		jsii.String(fmt.Sprintf("sudo -u ec2-user git clone %s /tmp/hl-recorder-clone", repoURL)),
		jsii.String("sudo -u ec2-user mv /tmp/hl-recorder-clone/.git /tmp/hl-recorder-clone/* /tmp/hl-recorder-clone/.[!.]* /opt/hl-recorder/ 2>/dev/null || true"),
		jsii.String("rmdir /tmp/hl-recorder-clone 2>/dev/null || true"),

		// Create virtual environment and install dependencies
		jsii.String("/root/.local/bin/uv venv /opt/hl-recorder/.venv --python 3.12"),
		jsii.String("/root/.local/bin/uv pip install --python /opt/hl-recorder/.venv/bin/python -e /opt/hl-recorder"),

		// Set ownership for ec2-user
		jsii.String("chown -R ec2-user:ec2-user /opt/hl-recorder"),

		// Create systemd service
		jsii.String(`cat > /etc/systemd/system/hl-recorder.service << 'SERVICEEOF'
[Unit]
Description=Hyperliquid Market Data Recorder
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/opt/hl-recorder
ExecStart=/opt/hl-recorder/.venv/bin/hl-recorder --config config/symbols.yaml --data-root /data --log-file /data/logs/recorder.log
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
Environment="PYTHONUNBUFFERED=1"

[Install]
WantedBy=multi-user.target
SERVICEEOF`),

		// Enable and start the service
		jsii.String("systemctl daemon-reload"),
		jsii.String("systemctl enable hl-recorder.service"),
		jsii.String("systemctl start hl-recorder.service"),
	)

	// Create EBS volume for data storage (10GB gp3)
	dataVolume := awsec2.NewVolume(stack, jsii.String("DataVolume"), &awsec2.VolumeProps{
		AvailabilityZone: jsii.String(fmt.Sprintf("%sa", *props.Env.Region)), // Tokyo ap-northeast-1a
		Size:             awscdk.Size_Gibibytes(jsii.Number(10)),
		VolumeType:       awsec2.EbsDeviceVolumeType_GP3,
		RemovalPolicy:    awscdk.RemovalPolicy_SNAPSHOT,
	})

	// EC2 Instance
	instance := awsec2.NewInstance(stack, jsii.String("HLRecorderInstance"), &awsec2.InstanceProps{
		InstanceType: awsec2.NewInstanceType(jsii.String("t4g.micro")), // ARM/Graviton - 1GB RAM
		MachineImage: awsec2.MachineImage_LatestAmazonLinux2023(&awsec2.AmazonLinux2023ImageSsmParameterProps{
			CpuType: awsec2.AmazonLinuxCpuType_ARM_64,
		}),
		Vpc: vpc,
		VpcSubnets: &awsec2.SubnetSelection{
			SubnetType: awsec2.SubnetType_PUBLIC,
		},
		SecurityGroup:            sg,
		Role:                     role,
		UserData:                 userData,
		AssociatePublicIpAddress: jsii.Bool(true),
		InstanceName:             jsii.String("hl-recorder"),
		BlockDevices: &[]*awsec2.BlockDevice{
			{
				DeviceName: jsii.String("/dev/xvda"),
				Volume: awsec2.BlockDeviceVolume_Ebs(jsii.Number(8), &awsec2.EbsDeviceOptions{
					VolumeType: awsec2.EbsDeviceVolumeType_GP3,
				}),
			},
		},
	})

	// Add key pair if SSH is enabled
	if props.EnableSSH && props.KeyPair != "" {
		instance.Instance().AddPropertyOverride(jsii.String("KeyName"), props.KeyPair)
	}

	// Attach data volume
	awsec2.NewCfnVolumeAttachment(stack, jsii.String("DataVolumeAttachment"), &awsec2.CfnVolumeAttachmentProps{
		Device:     jsii.String("/dev/sdf"),
		InstanceId: instance.InstanceId(),
		VolumeId:   dataVolume.VolumeId(),
	})

	// Outputs
	awscdk.NewCfnOutput(stack, jsii.String("InstanceID"), &awscdk.CfnOutputProps{
		Value:       instance.InstanceId(),
		Description: jsii.String("EC2 instance ID for SSM commands"),
		ExportName:  jsii.String("HLRecorderInstanceID"),
	})

	awscdk.NewCfnOutput(stack, jsii.String("InstancePublicIP"), &awscdk.CfnOutputProps{
		Value:       instance.InstancePublicIp(),
		Description: jsii.String("EC2 public IP address"),
	})

	awscdk.NewCfnOutput(stack, jsii.String("DataVolumeID"), &awscdk.CfnOutputProps{
		Value:       dataVolume.VolumeId(),
		Description: jsii.String("EBS data volume ID"),
	})

	return stack
}
