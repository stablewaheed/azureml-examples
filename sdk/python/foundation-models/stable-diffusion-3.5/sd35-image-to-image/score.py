# Copyright 2024 Stability AI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Score file for making inferences against Stable Diffusion inside an Azure Endpoint
"""

import io
import os
import logging
import torch

from diffusers import AutoPipelineForImage2Image
from PIL import Image

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global pipe
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "stable-diffusion-3.5-medium"
    )

    pipe = AutoPipelineForImage2Image.from_pretrained(
        model_path, torch_dtype=torch.float16
    )
    pipe.to('cuda')

    logging.info("Init complete")


def run(request):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    logging.info("Request received")

    # For Stable Diffusion 3.5 image-to-image the POST request data is NOT "Content-Type: application/json",
    # it is "Content-Type: multipart/form-data", with request.form and request.file objects

    image_file = request.files['file']
    init_image = Image.open(image_file.stream).resize((1024, 1024))

    strength = float(request.form['strength'])
    guidance_scale = float(request.form['guidance_scale'])

    images = pipe(
        prompt=request.form['prompt'],
        image=init_image,
        strength=strength,
        guidance_scale=guidance_scale,
        negative_prompt=request.form['negative_prompt']
    ).images

    image_bytes = io.BytesIO()
    images[0].save(image_bytes, 'PNG')
    image_bytes.seek(0)
    
    logging.info("Request processed")
    return image_bytes