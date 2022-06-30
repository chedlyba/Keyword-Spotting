# Pull nginx image
FROM nginx

# Remove the default config file
RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d
