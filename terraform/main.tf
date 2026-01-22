# terraform/main.tf
provider "tencentcloud" {
  # 优先使用变量（通过 TF_VAR_secret_id 设置），如果为空则使用环境变量
  # 如果变量为空字符串，Terraform 会使用 null，provider 会自动回退到环境变量
  secret_id  = var.secret_id != "" ? var.secret_id : null
  secret_key = var.secret_key != "" ? var.secret_key : null
  region     = var.region
}

# 查找 Ubuntu 22.04 镜像（兼容指定实例类型）
data "tencentcloud_images" "ubuntu" {
  image_type    = ["PUBLIC_IMAGE"]
  os_name       = "ubuntu"
  instance_type = var.instance_type  # 确保镜像与实例类型兼容
}

# 打包配置目录用于上传
data "archive_file" "config" {
  type        = "zip"
  output_path = "${path.module}/.terraform-config.zip"
  source_dir  = "${path.module}"
}

# VPC
resource "tencentcloud_vpc" "main" {
  name       = "quant-vpc"
  cidr_block = "10.0.0.0/16"
}

# Subnet
resource "tencentcloud_subnet" "main" {
  vpc_id            = tencentcloud_vpc.main.id
  name              = "quant-subnet"
  cidr_block        = "10.0.1.0/24"
  availability_zone = var.availability_zone
}

# Security Group
resource "tencentcloud_security_group" "quant_sg" {
  name        = "quant-sg"
  description = "Security group for quant server"
}

# Allow SSH from your IP (optional: replace with your IP)
resource "tencentcloud_security_group_rule" "ssh" {
  security_group_id = tencentcloud_security_group.quant_sg.id
  type              = "ingress"
  cidr_ip           = "0.0.0.0/0"  # 🔐 建议替换为你的固定 IP，如 "203.0.113.1/32"
  ip_protocol       = "tcp"
  port_range        = "22"
  policy            = "accept"
}

# Note: SQLite is file-based, no network ports needed

# Deny all other inbound (使用多个规则，因为不能使用 port_range "0-65535" 和 ip_protocol "all" 的组合)
# 注意：腾讯云安全组默认拒绝所有流量，这个规则可能不需要
# 如果需要明确拒绝，可以创建多个规则覆盖主要端口范围

# 生成 /etc/default/filebeat（含密钥）
resource "local_file" "filebeat_env" {
  content = templatefile("${path.module}/templates/filebeat-env.tpl", {
    cls_secret_id  = var.cls_secret_id
    cls_secret_key = var.cls_secret_key
    cls_topic_id   = var.cls_topic_id
  })
  filename = "${path.module}/filebeat.env"
}

# CVM Instance
resource "tencentcloud_instance" "quant_server" {
  instance_name     = "quant-engine"
  availability_zone = var.availability_zone
  # 如果指定了 image_id 则使用，否则使用第一个可用的 Ubuntu 镜像（兼容实例类型）
  image_id = var.image_id != "" ? var.image_id : data.tencentcloud_images.ubuntu.images[0].image_id
  instance_type     = var.instance_type
  system_disk_type  = "CLOUD_SSD"
  system_disk_size  = 50

  data_disks {
    data_disk_type = "CLOUD_PREMIUM"
    data_disk_size = var.data_disk_size  # 256 GB
  }

  vpc_id                 = tencentcloud_vpc.main.id
  subnet_id              = tencentcloud_subnet.main.id
  orderly_security_groups = [tencentcloud_security_group.quant_sg.id]

  # 如果设置了带宽，需要分配公网 IP
  internet_max_bandwidth_out = var.internet_bandwidth  # 调试设为 1，生产设为 0
  allocate_public_ip         = var.internet_bandwidth > 0 ? true : false

  # 自动运行初始化脚本
  user_data = base64encode(templatefile("${path.module}/init.sh", {
    ssh_public_key = var.ssh_public_key != "" ? var.ssh_public_key : file("${pathexpand(var.ssh_private_key)}.pub")
  }))

  tags = {
    app_name = "alpha-sentinel-bot"
    environment = "prod"
    managed_by = "hansen"
  }

  # 通过 provisioner 上传配置（需 SSH）
  connection {
    type        = "ssh"
    host        = self.public_ip
    user        = "ubuntu"
    private_key = file(pathexpand(var.ssh_private_key))
    timeout     = "5m"
    agent       = false
  }

  # 上传打包的配置文件
  provisioner "file" {
    source      = data.archive_file.config.output_path
    destination = "/tmp/terraform-config.zip"
  }

  provisioner "file" {
    source      = "${path.module}/filebeat.env"
    destination = "/tmp/filebeat.env"
  }

  provisioner "remote-exec" {
    inline = [
      # 等待系统完全启动
      "sleep 5",
      # 验证文件已上传
      "test -f /tmp/terraform-config.zip && echo '✅ ZIP 文件已上传' || (echo '❌ ZIP 文件不存在' && exit 1)",
      "test -f /tmp/filebeat.env && echo '✅ filebeat.env 已上传' || (echo '❌ filebeat.env 不存在' && exit 1)",
      # 安装 unzip（如果不存在）
      "which unzip >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y unzip)",
      # 解压配置文件（只解压需要的目录）
      "cd /tmp && unzip -q -o terraform-config.zip 'systemd/*' 'monitoring/*' 'logging/*' -d .",
      # 验证文件已解压
      "test -d /tmp/systemd && echo '✅ systemd 目录已解压' || (echo '❌ systemd 目录不存在' && exit 1)",
      "test -d /tmp/monitoring && echo '✅ monitoring 目录已解压' || (echo '❌ monitoring 目录不存在' && exit 1)",
      "test -d /tmp/logging && echo '✅ logging 目录已解压' || (echo '❌ logging 目录不存在' && exit 1)",
      # 执行初始化
      "sudo cp /tmp/filebeat.env /etc/default/filebeat",
      "sudo chmod 600 /etc/default/filebeat",
      "sudo /tmp/init.sh"
    ]
  }
}
