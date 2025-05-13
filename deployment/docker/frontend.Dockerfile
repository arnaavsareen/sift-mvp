FROM node:16-alpine as build

WORKDIR /app

# Copy package.json and package-lock.json first for better caching
COPY frontend/package*.json ./

# Install dependencies
RUN npm install

# Install HLS.js for video streaming
RUN npm install hls.js

# Copy frontend files
COPY frontend/ .

# Build the app
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built files from the build stage
COPY --from=build /app/build /usr/share/nginx/html

# Copy nginx config
COPY deployment/nginx/default.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 3000

# Command to run nginx
CMD ["nginx", "-g", "daemon off;"]