# Multi-stage Dockerfile for cachealign-proxy
FROM rust:1.80-alpine as builder

WORKDIR /usr/src/cachealign-proxy
RUN apk add --no-cache musl-dev pkgconfig openssl-dev

COPY Cargo.toml ./
COPY src ./src

RUN cargo build --release

# Minimal Runtime Container
FROM alpine:3.20

WORKDIR /app
RUN apk add --no-cache ca-certificates tzdata

COPY --from=builder /usr/src/cachealign-proxy/target/release/cachealign-proxy /usr/local/bin/cachealign-proxy

ENV PORT=8080
EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/cachealign-proxy"]