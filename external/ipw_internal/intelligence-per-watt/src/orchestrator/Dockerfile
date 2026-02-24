FROM nginx:latest
RUN echo "<html><body>Hello John</body></html>" > /usr/share/nginx/html/index.html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]