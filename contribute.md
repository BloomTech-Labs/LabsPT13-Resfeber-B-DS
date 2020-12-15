# Contribution
General notes and pointers for contributing.

## Environment Variables
The following environment variables are needed to run the api
```
MAPBOX_TOKEN=urmapboxtokenhere
```
Best practice is to create a `.env` file in the main directory, and let 
docker-compose handle importing the environment variables for local deployments. 
`.env` is ignored for elastic beanstalk, and environment variables will need to 
be set up on the eb console.

## References
[AWS Elastic Beanstalk environment variables](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/create_deploy_docker.container.console.html#docker-env-cfg.env-variables)  
[docker-compose environment variables](https://docs.docker.com/compose/environment-variables/)  