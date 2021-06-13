## How to run the project with Docker

1. Open terminal in the repository root

2. Build docker image:
`docker build --tag yelp .`
   
3. Run container from an image with mounted volume. 
   Instead of `{desired_host_dir}` pass absolute path to the directory where an output will be stored
`docker run -v {desired_host_dir}:/app/output yelp`
   
