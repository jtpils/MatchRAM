#!/bin/bash
docker run -it -p 8888:8888 -v $(pwd):/notebooks/ siavashk/siemens:cpu
