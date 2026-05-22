FROM nvcr.io/nvidia/rapidsai/base:26.04-cuda12-py3.13

WORKDIR /workspace

# scikit-image is needed for the real-image demos (human_mitosis,
# immunohistochemistry, binary_blobs). It comes pre-installed in some RAPIDS
# base images but we pin it here so the build is self-contained. pooch is
# required for skimage.data.human_mitosis (fetches mitosis.tif on first use).
RUN pip install --no-cache-dir scikit-image pooch

COPY install_verification.py .
COPY hello_cucim.py .
COPY relevant_uses.py .
COPY hello_cucim_blobs.py .
COPY hello_cucim_mitosis.py .
COPY hello_cucim_ihc.py .

CMD ["python", "install_verification.py"]
