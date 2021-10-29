## Symptoms-based predictive models of the COVID-19 disease in children

This is the code used for the preprocessing, training, evaluation, fine-tuning and feature importance 
extraction steps for the paper

"Symptoms-based predictive models of the COVID-19 disease in children".

The code is provided in a script form in the `src` folder and in a step-by-step basis in the `notebooks` folder. The
former is great for more easily understanding how the pipelines work and are structured. The latter is awesome for following
along the whole process, with intermediate explanations, especially for a **non-technical** public. To open the notebooks
you will need a specific software, such as `jupyter`.

The run the pipelines from the script files, install the requirements in a new environment:

```
pip3 install -r requirements.txt
```
and write down the paths for the data, figures and results that are to be created
from the process: the path variables to be overwritten are found in the corresponding `configs.py` files.
Finally, just run the corresponding `main.py` file in `preprocessing` and `modeling`,
respectively:

```
python3 main.py
```

Note that we do **not** provide the data with this repository. Instead, this software is provided for
results transparency purposes.

<p align="center">
    <a href="https://www.copedicat.cat/">
        <img src="https://pbs.twimg.com/profile_images/1309527192381067270/6zbTWN-M_400x400.jpg" alt="drawing" width="50"/>
    </a>
    <a href="https://biocomsc.upc.edu/en">
        <img src="https://biocomsc.upc.edu/en/Logo_2014SGR1093_BIOCOMSC.jpg" alt="drawing" width="90"/>
    </a> 
    <a href="https://www.upc.edu/en">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/Logo_UPC.svg/2048px-Logo_UPC.svg.png" alt="drawing" width="50"/>
    </a>
</p>