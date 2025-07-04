# Docker 基础操作命令笔记

## 1. 容器管理

### 查看所有容器（包括已停止）

```bash
docker ps -a
```

### 查看正在运行的容器

```bash
docker ps
```

### 启动容器

```bash
docker start <容器名或ID>
```

### 停止容器

```bash
docker stop <容器名或ID>
```

### 删除容器

```bash
docker rm <容器名或ID>
```

### 强制删除正在运行的容器

```bash
docker rm -f <容器名或ID>
```

### 进入容器交互式终端

```bash
docker exec -it <容器名或ID> /bin/bash
```

---

## 2. 镜像管理

### 查看本地所有镜像

```bash
docker images
```

### 拉取镜像

```bash
docker pull <镜像名>:<标签>
# 例如：docker pull heartexlabs/label-studio:latest
```

### 删除镜像

```bash
docker rmi <镜像名或ID>
```

---

## 3. 端口与网络

### 查看容器端口映射

```bash
docker ps
# PORTS 列显示本地端口与容器端口的映射关系
```

### 查询本机端口占用情况

```bash
lsof -i :8080
# 或
netstat -anp tcp | grep 8080
```

---

## 4. 容器切换与重建

### 切换（重启）到新镜像版本

```bash
# 1. 停止并删除旧容器
# 2. 启动新容器（可用新镜像/新参数）
docker stop <旧容器名>
docker rm <旧容器名>
docker run -d --name <新容器名> -p <本地端口>:<容器端口> <镜像名>:<标签>
```

---

## 5. 日志与排查

### 查看容器日志

```bash
docker logs <容器名或ID>
```

### 实时查看日志输出

```bash
docker logs -f <容器名或ID>
```

---

## 6. 其他常用命令

### 查看容器详细信息

```bash
docker inspect <容器名或ID>
```

### 查看镜像详细信息

```bash
docker inspect <镜像名或ID>
```

---

> 本笔记适用于日常开发环境下的 Docker 基础操作，适合快速查阅和实操。
