# third_party

Find below the existing configs for the third_party folder. To have your YAML file indexed, add a docstring at the beginning of the file.
The docstring should be a series of comment lines starting with two '#' characters.

Example:
```
## This is a docstring
## describing the YAML file.
key: value
```


## amazon_all_beauty.yaml

Config file for preprocessing datasets from Amazon Reviews 2023 for the All_Beauty domain.
Dataset details: https://amazon-reviews-2023.github.io/index.html


## amazon_default.yaml

Default Config file for preprocessing datasets from Amazon Reviews 2023.
This config is meant to be extended by other config files for specific versions of the Amazon dataset.
In particular, the user must specify the domains parameter in the extending config file.
Dataset details: https://amazon-reviews-2023.github.io/index.html


## amazon_magazine_subscriptions.yaml

Config file for preprocessing datasets from Amazon Reviews 2023 for the Magazine_Subscriptions domain.
Dataset details: https://amazon-reviews-2023.github.io/index.html
