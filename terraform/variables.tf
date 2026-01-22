variable "region" {
  description = "Tencent Cloud region"
  default     = "ap-tokyo"
}

variable "availability_zone" {
  description = "Availability zone"
  default     = "ap-tokyo-2"  # 使用 ap-tokyo-2 避免库存不足
}

variable "image_id" {
  description = "Ubuntu 24.04 image ID (will be auto-detected if not set)"
  default     = ""   # 留空，通过 data source 自动查找
}


variable "instance_type" {
  description = "CVM type: 2vCPU / 4GB"
  default     = "S5.MEDIUM4"      # 标准型 S5，2核4G（S5.SMALL2 库存不足时使用）
}

variable "data_disk_size" {
  description = "Data disk size in GB"
  default     = 256              # 256GB
}

variable "internet_bandwidth" {
  description = "Public bandwidth (Mbps). Set to 0 in production."
  default     = 1
}

# 可选：如果环境变量不工作，可以显式指定
variable "secret_id" {
  description = "Tencent Cloud secret ID (optional, defaults to TENCENTCLOUD_SECRET_ID env var)"
  type        = string
  sensitive   = true
  default     = ""
}

variable "secret_key" {
  description = "Tencent Cloud secret key (optional, defaults to TENCENTCLOUD_SECRET_KEY env var)"
  type        = string
  sensitive   = true
  default     = ""
}

variable "cls_secret_id" {
  description = "CLS (Cloud Log Service) secret ID for Filebeat. Can be set via TF_VAR_cls_secret_id or CLS_SECRET_ID env var"
  type        = string
  sensitive   = true
  default     = ""  # 默认空值，可通过环境变量或 terraform.tfvars 设置
}

variable "cls_secret_key" {
  description = "CLS (Cloud Log Service) secret key for Filebeat. Can be set via TF_VAR_cls_secret_key or CLS_SECRET_KEY env var"
  type        = string
  sensitive   = true
  default     = ""  # 默认空值，可通过环境变量或 terraform.tfvars 设置
}

variable "cls_topic_id" {
  description = "CLS (Cloud Log Service) topic ID for Filebeat. Can be set via TF_VAR_cls_topic_id or CLS_TOPIC_ID env var"
  type        = string
  default     = ""  # 默认空值，可通过环境变量或 terraform.tfvars 设置
}

variable "ssh_private_key" {
  description = "Path to SSH private key for provisioning (supports ~ expansion)"
  type        = string
  default     = "~/.ssh/id_ed25519"  # 使用 ed25519 密钥（如果存在）
}

variable "ssh_public_key" {
  description = "SSH public key content (auto-read from private key path if not set)"
  type        = string
  default     = ""
}
