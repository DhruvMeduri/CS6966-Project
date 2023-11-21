# Copyright 2022 The HuggingFace Datasets Authors.
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
"""FairFace loading script."""


import csv
import os

import datasets

_CITATION = """\
@inproceedings{karkkainenfairface,
    title={FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation},
    author={Karkkainen, Kimmo and Joo, Jungseock},
    booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
    year={2021},
    pages={1548--1558}
}
"""

_DESCRIPTION = """\
FairFace is a face image dataset which is race balanced. It contains 108,501 images from 7 different race groups: White, Black, Indian, East Asian, Southeast Asian, Middle Eastern, and Latino.
Images were collected from the YFCC-100M Flickr dataset and labeled with race, gender, and age groups.
"""

_HOMEPAGE = "https://github.com/joojs/fairface"

_LICENSE = "CC BY 4.0"

_URLS = {
    "labels": {
        "train": "https://drive.google.com/u/0/uc?id=1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH&export=download",
        "val": "https://drive.google.com/u/0/uc?id=1wOdja-ezstMEp81tX1a-EYkFebev4h7D&export=download",
    },
    "images": {
        "0.25": "https://drive.google.com/u/0/uc?id=1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86&export=download",
        "1.25": "https://drive.google.com/u/0/uc?id=1g7qNOZz9wC7OfOhcPqH1EZ5bk1UFGmlL&export=download",
    },
}

_AGE_NAMES = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]
_GENDER_NAMES = ["Male", "Female"]
_RACE_NAMES = ["East Asian", "Indian", "Black", "White", "Middle Eastern", "Latino_Hispanic", "Southeast Asian"]

class TextCapsDataset(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="0.25", version=VERSION, description="Padding = 0.25"),
        datasets.BuilderConfig(name="1.25", version=VERSION, description="Padding = 1.25"),
    ]

    DEFAULT_CONFIG_NAME = "0.25"

    def _info(self):
        features = datasets.Features(
            {
                "image": datasets.Image(),
                "age": datasets.ClassLabel(names=_AGE_NAMES),
                "gender": datasets.ClassLabel(names=_GENDER_NAMES),
                "race": datasets.ClassLabel(names=_RACE_NAMES),
                "service_test": datasets.Value("bool"), # NOT SURE WHAT THIS IS YET
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(_URLS)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "images_folder": data_dir["images"],
                    "labels": data_dir["labels"]["train"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "images_folder": data_dir["images"],
                    "labels": data_dir["labels"]["val"],
                },
            ),
        ]

    def _generate_examples(
        self, images_folder, labels,
    ):
        with open(labels, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(
                csv_file, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True
            )
            next(csv_reader)
            counter = 0
            for row in csv_reader:
                file, age, gender, race, service_test = row
                image_path = os.path.join(images_folder[self.config_id], file)
                yield counter, {
                    "image": image_path,
                    "age": age,
                    "gender": gender,
                    "race": race,
                    "service_test": True if service_test == "True" else False,
                }
                counter += 1
