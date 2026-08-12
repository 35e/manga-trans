# The dashboard: built once, then served as static files by nginx.
FROM node:22-alpine AS build

WORKDIR /app

# pnpm comes with node as a corepack shim; pinning it here rather than installing
# it separately keeps the lockfile and the tool that wrote it in step.
RUN corepack enable

# The lockfile alone first, so a change to the source does not send the install
# back to the network.
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

COPY . .

# Empty rather than unset: `lib/api.ts` falls back to http://localhost:8000 only
# when this is nullish, and an empty base makes every call a same-origin one that
# nginx proxies through to the API. That is what takes CORS out of the picture.
ENV VITE_API_URL=""
RUN pnpm build


FROM nginx:1.29-alpine AS serve

COPY --from=build /app/dist /usr/share/nginx/html

# A template rather than a finished config: the image's entrypoint fills
# ${NGINX_LOCAL_RESOLVERS} in from the container's own /etc/resolv.conf, which is
# what lets the API's address be looked up again after it restarts. The filter
# keeps envsubst off everything else, so nginx's own $host, $remote_addr and the
# rest survive into the config.
COPY default.conf.template /etc/nginx/templates/default.conf.template
# The entrypoint only works the resolvers out when asked to; without this it
# returns early and the template is filled with an empty string, which nginx
# rejects at start with "host not found in resolver".
ENV NGINX_ENTRYPOINT_LOCAL_RESOLVERS=1 \
    NGINX_ENVSUBST_FILTER=NGINX_LOCAL_RESOLVERS

EXPOSE 80
