from setuptools import setup, find_packages

setup(
    name="amharicNLP",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'nltk>=3.6',
        'regex>=2021.11'
    ],
    package_data={
        'amharicNLP': ['resources/*.txt']
    },
    author="yonas abebe",
    author_email="abebeyonas88@gmail.com",
    description="Amharic NLP preprocessing toolkit",
    url="https://github.com/yonasab12/amharicNLP",
)