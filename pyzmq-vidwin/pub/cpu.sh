#!/bin/sh
python DockerStatsWriter.py &
docker-compose -f docker-compose-cpu.yml up