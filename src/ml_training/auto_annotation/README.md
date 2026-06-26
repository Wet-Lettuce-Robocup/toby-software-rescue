When using CVAT, you can automatically annotate images using a ml model. How I did this:

In the directory cvat/serverless, paste the shell script and 'yolov8' directory, then run:

 `docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d`
` docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml -f docker-compose.override.yml up -d`


`./serverless/deploy_cpu.sh serverless/yolov8/nuclio`

If it works, the model should appear in the 'models' tab of CVAT. To make it work on mac, I had do the following:
`brew install bash`
`brew install coreutils`