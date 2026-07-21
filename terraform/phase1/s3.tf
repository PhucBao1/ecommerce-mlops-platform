resource "aws_s3_bucket" "lakehouse" {

  bucket        = "${var.project_name}-lakehouse"
  force_destroy = true

}

resource "aws_s3_bucket_versioning" "lakehouse" {

  bucket = aws_s3_bucket.lakehouse.id

  versioning_configuration {
    status = "Enabled"
  }
}
