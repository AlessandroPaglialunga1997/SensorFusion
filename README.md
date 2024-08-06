# UKF on Manifold

Implementation of UKF on Manifold, at the moment fuse IMU or Odometry with GPS (global positioning). Look at `ukf_manifolds_notes.pdf` for the theoretical part.  

### Installing dependencies (eigen3 and GTest)
```sudo apt install libeigen3-dev libgtest-dev```


### Build
In pkg folder run
```mkdir build && cd build && cmake ..```

```make -j```

### Run
We provide some examples with syntethic data and an example using KITTI dataset (fusing IMU and GPS)

```cd build/examples```

When you run an example just specify the path where to write the estimate file, for instance `/home/username/Desktop/`, the filename is given by default.

```./example_imu_gps_synthetic_data path-to-estimate```

All the executables need to be run in the same way!

### Visualize data
The data ouput follows the following format \<tx ty tz>, only the global position of the body/robot/car is written in the text file. You can visualize this with you favourite plotter. A stupid way can be using `gnuplot`. Once installed, `cd` in the folder where you dumped the files and run:

```gnuplot```

```splot "traj_est.txt" using 1:2:3 w l, "traj_gt.txt" using 1:2:3 w l``` 

Note for KITTI example we just dump the estimate, therefore to visualize in `gnuplot`:

```plot "kitti_estimate.txt" using 1:2 w l``` 

This is just the planar motion, note difference with `plot/splot`, you can visualize in 3D but you need to fix the aspect ratio of the plot, since the "z" component has much less drift compared to the other axis, however, if you don't fix the aspect ratio the estimate seems fucked up!
