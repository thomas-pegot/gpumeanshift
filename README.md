gpumeanshift
============

Python binding OpenCV gpu::meanshift with Boost.Numpy library.


How to use:
-----------

-download and build:

```
git clone https://github.com/thomas-pegot/gpumeanshift.git
cd gpumeanshift
make
```

-launch python:

```python
from gpumeanshift import filter

import cv2

I = cv2.imread("data/star.jpg", cv2.IMREAD_COLOR)

Out = filter(I, 12, 12)

cv2.imshow("result",Out)
```
