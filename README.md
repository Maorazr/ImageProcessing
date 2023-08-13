
<div align="center">
<h1 align="center">
<img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" width="100" />
<br>ImageProcessing
</h1>
<h3>◦ Transforming pixels with precision</h3>
<h3>◦ Developed with the software and tools listed below.</h3>

<p align="center">
<img src="https://img.shields.io/badge/Python-3776AB.svg?style&logo=Python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/JSON-000000.svg?style&logo=JSON&logoColor=white" alt="JSON" />
</p>

</div>




## 📍 Overview

The ImageProcessing project is a Python application that offers a range of functionalities for image processing and text annotation. It provides a user-friendly graphical interface for loading and annotating images, utilizing the capabilities of libraries like OpenCV, NumPy, and PyTorch. The project's core functionalities include character recognition using a neural network, cropping letters from images, and generating words based on recognized characters. Overall, it aims to simplify the task of image processing and make text annotation more efficient.

---

## ⚙️ Features

| Feature                | Description                           |
| ---------------------- | ------------------------------------- |
| **⚙️ Architecture**     | The codebase follows a modular architecture, with separate files for different functionalities. It utilizes a GUI for image processing and text annotation. The system follows a simple and straightforward design. |
| **📖 Documentation**   | The documentation is limited and could be improved. There is a need for more detailed comments and function-level documentation to enhance code comprehension. |
| **🔗 Dependencies**    | The system relies on external libraries like OpenCV, NumPy, and PyTorch for image processing, character recognition, and GUI functionality. All the dependencies are clearly defined and easy to install. |
| **🧩 Modularity**      | The codebase is organized into separate files for different functionalities – main.py, gui.py, and utils.py. Each file focuses on a specific aspect of image processing and text annotation and can be easily maintained or extended. |
| **✔️ Testing**          | There is no mention of specific testing strategies or tools being used in the codebase. The absence of tests can impact long-term stability and maintainability. Implementing tests would be beneficial for code quality. |
| **⚡️ Performance**      | Given the nature of the system, performance may vary depending on the size and complexity of the image being processed. However, since the codebase utilizes popular and efficient libraries for image processing, it should generally perform satisfactorily. |
| **🔐 Security**        | The codebase does not appear to have specific security measures since its primary focus is on image processing and text annotation. It may have limited security vulnerabilities related to file access, but that would depend on the server environment in which it is deployed. |
| **🔀 Version Control** | The codebase utilizes Git for version control and is hosted on GitHub. While it is not explicitly mentioned how the repository is managed, Git provides features like branching, tagging, and merge management that can be utilized for effective version control. |
| **🔌 Integrations**    | The system does not have explicit integrations with other systems or services based on the information provided. It is a stand-alone image processing and text annotation tool. |
| **📶 Scalability**     | It is unclear from the provided information how well the system can handle growth. Factors like performance optimization, parallelization, and distributed processing should be considered to assess scalability. More information would be needed to evaluate it effectively. |

---


## 🧩 Modules

<details closed><summary>Src</summary>

| File                                                                            | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ---                                                                             | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| [main.py](https://github.com/Maorazr/ImageProcessing/blob/main/src/main.py)     | The code imports the required modules,'OrderedDict' from'collections', and all the functions from a module called'utils'. It defines a function called'main', which calls a function'gui'. The'main' function is executed only if the script is run directly as the main program.                                                                                                                                                                   |
| [gui.py](https://github.com/Maorazr/ImageProcessing/blob/main/src/gui.py)       | The code snippet is implementing a function that sorts an array of integers. It uses a popular sorting algorithm called bubble sort, where it repeatedly compares adjacent elements and swaps them if they are in the wrong order.                                                                                                                                                                                                                  |
| [utiils.py](https://github.com/Maorazr/ImageProcessing/blob/main/src/utiils.py) | This code snippet defines a Tech Lead class and implements functions for image processing and text annotation. It includes a neural network model for character recognition, methods for cropping letters from images and generating words from the recognized characters. It also includes GUI functionality for loading and annotating images either by clicking or typing words. The code utilizes libraries such as OpenCV, NumPy, and PyTorch. |

</details>

---

### 📦 Installation

1. Clone the ImageProcessing repository:
```sh
git clone https://github.com/Maorazr/ImageProcessing
```

2. Change to the project directory:
```sh
cd ImageProcessing
```

3. Install the dependencies:
```sh
pip install -r requirements.txt
```

### 🎮 Using ImageProcessing

```sh
python main.py
```

