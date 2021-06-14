## How to run the project with Docker

1. Open terminal in the repository root folder

2. Build docker image:
   `docker build --tag yelp .`
   
3. Run container from an image with mounted volume:
   
   `docker run -v {host_output_dir}:/app/output --env-file remote_urls.env yelp`
   `{host_output_dir}` - absolute path to the directory where an output will be stored.
   
   This will download approx. 2.7 Gb of data and may take a while

4. Results (plots, images) will be saved in the `{host_output_dir}` directory
   
