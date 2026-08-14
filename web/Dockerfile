# The dashboard: built once, then served as static files by nginx.
FROM node:22-alpine AS build

WORKDIR /app

RUN corepack enable

# The lockfile alone first, so a source change does not re-run the install.
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

COPY . .

# Empty rather than unset: `lib/api.ts` falls back only when this is nullish, and
# an empty base is what makes every call same-origin.
ENV VITE_API_URL=""
RUN pnpm build


FROM nginx:1.29-alpine AS serve

COPY --from=build /app/dist /usr/share/nginx/html

# A template, so the entrypoint can fill ${NGINX_LOCAL_RESOLVERS} in from the
# container's own /etc/resolv.conf. The filter keeps envsubst off nginx's own
# $host and $remote_addr.
COPY default.conf.template /etc/nginx/templates/default.conf.template
# Without this the entrypoint returns early and nginx refuses to start with
# "host not found in resolver".
ENV NGINX_ENTRYPOINT_LOCAL_RESOLVERS=1 \
    NGINX_ENVSUBST_FILTER=NGINX_LOCAL_RESOLVERS

EXPOSE 80
