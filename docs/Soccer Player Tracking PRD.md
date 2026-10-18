Product Requirements Document

**Soccer Player Tracking & Movement Analytics**

| Field | Detail |
| :---- | :---- |
| Document owner | Kazi Hossain |
| Team Members | Abeih Humani, David Girgis, Ratul Deoun |
| Target timeline | 8 weeks |

# **1\. Overview**

This project builds a computer-vision pipeline that detects and tracks soccer players in match images and video, converts their positions from pixel space to coordinates, and estimates each player's speed and total distance covered. Players are also grouped into teams using jersey-color clustering.

## **1.1 Problem Statement**

Player movement data (sprint speed, distance covered, positioning) is valuable for performance analysis and coaching. This project is a way to help extract those metrics.

# **2\. Goals and Non-Goals**

## **2.1 Goals**

* Detect and track all players in each frame of soccer video with high accuracy

* Map pixel coordinates to real pitch coordinates 

* Estimate per-player speed (km/h and m/s) and cumulative distance covered (meters / kilometers)

* Automatically assign players (including goalkeepers) to teams based on jersey color, with a separate cluster for referees

* Evaluate the model using standard metrics MAE, MSE, and R².

## **2.2 Non-Goals**

* This works with recorded footage not real-time broadcast

* Advanced analytics such as formation detection or expected goals

# **3\. Users and Use Cases**

| User | Need | How this helps |
| :---- | :---- | :---- |
| Data science student | Movement metrics  | Generates speed and distance from public footage |
| Coach | Quick positioning insight | player distance and speed information |

# **4\. Functional Requirements**

| Requirement | Description |
| :---- | :---- |
| Player detection | Use a YOLO object detection model to detect players in every frame from soccer match video clips |
| Multi-object tracking | Assign IDs to players across frames. Link detections across frames into trajectories |
| Pitch points | Use OpenCV perspective transformation to convert image coordinates into real-world pitch coordinates |
| Speed and distance estimation | Compute player speed and cumulative distance from transformed positions and frame rate |
| Team assignment | Segment player by applying K-Means on jersey colors  |
| Output visualization | Render annotated video with bounding boxes, team colors, speed, and distance  |
| Hosting | Host the project in Vercel and cloud |

# **5\. Technical Approach**

## **5.1 Key Components**

* **YOLO Object Detection:** Pretrained or fine-tuned YOLO model for players and goalkeepers.

* **Perspective Transformation (OpenCV):** Maps image/video coordinates to real-world pitch coordinates using standard pitch dimensions (105 m x 68 m).

* **K-Means Clustering:** Segments player pixels by jersey color to separate teams including goalkeeper

* **PCA:** Reduces dimensionality of high-dimensional datasets for visualization

## **5.2 Data Cleaning and Preprocessing**

* Convert all categorical or text fields to numerical values using encoding techniques (**K-fold target encoding**)

* Detect outliers using probability methods

* Compute relative percentages where raw counts are not comparable across clips

* Normalize or standardize features before clustering

## **5.3 Choosing the Number of Clusters (k)**

* **Elbow method:** Pick a range of k values. For each k value, compute the within-cluster sum of squared distances to centroid and plot sums against k. Choose k at the elbow where decrease slows significantly.

* **Silhouette score:** For each k, measure how similar each point is to its own cluster versus other clusters. The score range from \-1 to 1 (better score).

* **Validation:** Use a validation set to confirm the optimal k.

* **Algorithm comparison:** Experiment with other clustering algorithms (e.g. DBSCAN) and compare against K-Means.

# **6\. Evaluation and Success Metrics**

| Metric | What it measures | Target  |
| :---- | :---- | :---- |
| MSE | Error in speed/distance. Penalizes large errors more heavily |  |
| R² | how well model's speed and distance estimates track the real values | \> 0.85 |
| Detection  | YOLO detection quality | \> 0.90 for players/goalkeepers |
| Team-assignment accuracy | Correct jersey-color clustering | \> 95% |
| Silhouette score | Clustering separation quality | \> 0.5 |

# **7\. Milestones**

| Phase | Deliverables | Target date |
| :---- | :---- | :---- |
| 1\. Planning | Find dataset, set up repo | Week 1 |
| 2\. Data prep | Clean data, preprocessing, outlier handling | Week 2 |
| 3\. Detection and tracking | YOLO detection and player tracking working end to end | Weeks 3-4 |
| 4\. Clustering | PCA visualization to understand nested plots, K-Means team assignment, elbow/silhouette analysis | Week 5 |
| 5\. Pitch mapping | Perspective transform and pitch-coordinate projection | Week 6 |
| 6\. Metrics and evaluation | MSE/R² evaluation | Week 7 |
| 7\. Polish and demo | video, github readme, vercel hosting  | Week 8 |

# **8\. Risks and Mitigations**

| Risk | Mitigation |
| :---- | :---- |
| Camera pans and zooms break fixed homography | Re-estimate homography per frame using detected pitch keypoints and line markings |
| Hard to validate accuracy for speed/distance | make a small sample |

