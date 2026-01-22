output "public_ip" {
  value = tencentcloud_instance.quant_server.public_ip
}

output "instance_id" {
  value = tencentcloud_instance.quant_server.id
}
