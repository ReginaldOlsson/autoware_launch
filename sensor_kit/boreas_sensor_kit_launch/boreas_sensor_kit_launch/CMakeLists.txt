cmake_minimum_required(VERSION 3.14)
project(boreas_sensor_kit_launch)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

install(PROGRAMS
  script/boreas_cloud_adapter.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_auto_package(INSTALL_TO_SHARE
  launch
)
