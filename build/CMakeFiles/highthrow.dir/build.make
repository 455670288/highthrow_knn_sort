# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/firefly/ljh/highthrow_2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/firefly/ljh/highthrow_2/build

# Include any dependencies generated for this target.
include CMakeFiles/highthrow.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/highthrow.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/highthrow.dir/flags.make

CMakeFiles/highthrow.dir/src/main.cc.o: CMakeFiles/highthrow.dir/flags.make
CMakeFiles/highthrow.dir/src/main.cc.o: ../src/main.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/firefly/ljh/highthrow_2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/highthrow.dir/src/main.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/highthrow.dir/src/main.cc.o -c /home/firefly/ljh/highthrow_2/src/main.cc

CMakeFiles/highthrow.dir/src/main.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/highthrow.dir/src/main.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/firefly/ljh/highthrow_2/src/main.cc > CMakeFiles/highthrow.dir/src/main.cc.i

CMakeFiles/highthrow.dir/src/main.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/highthrow.dir/src/main.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/firefly/ljh/highthrow_2/src/main.cc -o CMakeFiles/highthrow.dir/src/main.cc.s

CMakeFiles/highthrow.dir/src/sort.cc.o: CMakeFiles/highthrow.dir/flags.make
CMakeFiles/highthrow.dir/src/sort.cc.o: ../src/sort.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/firefly/ljh/highthrow_2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/highthrow.dir/src/sort.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/highthrow.dir/src/sort.cc.o -c /home/firefly/ljh/highthrow_2/src/sort.cc

CMakeFiles/highthrow.dir/src/sort.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/highthrow.dir/src/sort.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/firefly/ljh/highthrow_2/src/sort.cc > CMakeFiles/highthrow.dir/src/sort.cc.i

CMakeFiles/highthrow.dir/src/sort.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/highthrow.dir/src/sort.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/firefly/ljh/highthrow_2/src/sort.cc -o CMakeFiles/highthrow.dir/src/sort.cc.s

CMakeFiles/highthrow.dir/src/knnDetector.cc.o: CMakeFiles/highthrow.dir/flags.make
CMakeFiles/highthrow.dir/src/knnDetector.cc.o: ../src/knnDetector.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/firefly/ljh/highthrow_2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/highthrow.dir/src/knnDetector.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/highthrow.dir/src/knnDetector.cc.o -c /home/firefly/ljh/highthrow_2/src/knnDetector.cc

CMakeFiles/highthrow.dir/src/knnDetector.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/highthrow.dir/src/knnDetector.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/firefly/ljh/highthrow_2/src/knnDetector.cc > CMakeFiles/highthrow.dir/src/knnDetector.cc.i

CMakeFiles/highthrow.dir/src/knnDetector.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/highthrow.dir/src/knnDetector.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/firefly/ljh/highthrow_2/src/knnDetector.cc -o CMakeFiles/highthrow.dir/src/knnDetector.cc.s

CMakeFiles/highthrow.dir/src/kalmanBoxTracker.cc.o: CMakeFiles/highthrow.dir/flags.make
CMakeFiles/highthrow.dir/src/kalmanBoxTracker.cc.o: ../src/kalmanBoxTracker.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/firefly/ljh/highthrow_2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/highthrow.dir/src/kalmanBoxTracker.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/highthrow.dir/src/kalmanBoxTracker.cc.o -c /home/firefly/ljh/highthrow_2/src/kalmanBoxTracker.cc

CMakeFiles/highthrow.dir/src/kalmanBoxTracker.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/highthrow.dir/src/kalmanBoxTracker.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/firefly/ljh/highthrow_2/src/kalmanBoxTracker.cc > CMakeFiles/highthrow.dir/src/kalmanBoxTracker.cc.i

CMakeFiles/highthrow.dir/src/kalmanBoxTracker.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/highthrow.dir/src/kalmanBoxTracker.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/firefly/ljh/highthrow_2/src/kalmanBoxTracker.cc -o CMakeFiles/highthrow.dir/src/kalmanBoxTracker.cc.s

CMakeFiles/highthrow.dir/src/adjuster.cc.o: CMakeFiles/highthrow.dir/flags.make
CMakeFiles/highthrow.dir/src/adjuster.cc.o: ../src/adjuster.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/firefly/ljh/highthrow_2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/highthrow.dir/src/adjuster.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/highthrow.dir/src/adjuster.cc.o -c /home/firefly/ljh/highthrow_2/src/adjuster.cc

CMakeFiles/highthrow.dir/src/adjuster.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/highthrow.dir/src/adjuster.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/firefly/ljh/highthrow_2/src/adjuster.cc > CMakeFiles/highthrow.dir/src/adjuster.cc.i

CMakeFiles/highthrow.dir/src/adjuster.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/highthrow.dir/src/adjuster.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/firefly/ljh/highthrow_2/src/adjuster.cc -o CMakeFiles/highthrow.dir/src/adjuster.cc.s

# Object files for target highthrow
highthrow_OBJECTS = \
"CMakeFiles/highthrow.dir/src/main.cc.o" \
"CMakeFiles/highthrow.dir/src/sort.cc.o" \
"CMakeFiles/highthrow.dir/src/knnDetector.cc.o" \
"CMakeFiles/highthrow.dir/src/kalmanBoxTracker.cc.o" \
"CMakeFiles/highthrow.dir/src/adjuster.cc.o"

# External object files for target highthrow
highthrow_EXTERNAL_OBJECTS =

highthrow: CMakeFiles/highthrow.dir/src/main.cc.o
highthrow: CMakeFiles/highthrow.dir/src/sort.cc.o
highthrow: CMakeFiles/highthrow.dir/src/knnDetector.cc.o
highthrow: CMakeFiles/highthrow.dir/src/kalmanBoxTracker.cc.o
highthrow: CMakeFiles/highthrow.dir/src/adjuster.cc.o
highthrow: CMakeFiles/highthrow.dir/build.make
highthrow: ../libs/librknn_api/lib/librknnrt.so
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_stitching.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_alphamat.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_aruco.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_bgsegm.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_bioinspired.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_ccalib.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_dnn_objdetect.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_dnn_superres.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_dpm.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_face.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_freetype.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_fuzzy.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_hdf.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_hfs.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_img_hash.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_intensity_transform.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_line_descriptor.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_mcc.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_quality.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_rapid.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_reg.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_rgbd.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_saliency.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_shape.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_stereo.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_structured_light.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_superres.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_surface_matching.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_tracking.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_videostab.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_viz.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_xobjdetect.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_xphoto.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_highgui.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_datasets.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_plot.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_text.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_ml.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_phase_unwrapping.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_optflow.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_ximgproc.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_video.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_dnn.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_videoio.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_objdetect.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_calib3d.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_features2d.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_flann.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_photo.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.5.1
highthrow: /usr/lib/aarch64-linux-gnu/libopencv_core.so.4.5.1
highthrow: CMakeFiles/highthrow.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/firefly/ljh/highthrow_2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable highthrow"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/highthrow.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/highthrow.dir/build: highthrow

.PHONY : CMakeFiles/highthrow.dir/build

CMakeFiles/highthrow.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/highthrow.dir/cmake_clean.cmake
.PHONY : CMakeFiles/highthrow.dir/clean

CMakeFiles/highthrow.dir/depend:
	cd /home/firefly/ljh/highthrow_2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/firefly/ljh/highthrow_2 /home/firefly/ljh/highthrow_2 /home/firefly/ljh/highthrow_2/build /home/firefly/ljh/highthrow_2/build /home/firefly/ljh/highthrow_2/build/CMakeFiles/highthrow.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/highthrow.dir/depend
