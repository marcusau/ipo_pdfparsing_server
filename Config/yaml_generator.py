import dataclasses

#/usr/bin/env python
# -*- coding: utf-8 -*-
import os, pathlib, sys,click,logging,string,csv,json,tempfile,re

sys.path.append(os.getcwd())

parent_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(parent_path))

master_path = parent_path.parent
sys.path.append(str(master_path))

project_path = master_path.parent
sys.path.append(str(project_path))

import yaml2pyclass


class Config(yaml2pyclass.CodeGenerator):
    @dataclasses.dataclass
    class ExternalApiClass:
        sec_firm: str
        pdf: str
    
    @dataclasses.dataclass
    class ApiClass:
        @dataclasses.dataclass
        class RouteClass:
            main: str
        
        port: int
        route: RouteClass
    
    @dataclasses.dataclass
    class NerModelsClass:
        @dataclasses.dataclass
        class EngClass:
            model: str
            sent_vocab: str
            tag_vocab: str
            max_len: int
        
        eng: EngClass
    
    external_API: ExternalApiClass
    API: ApiClass
    ner_models: NerModelsClass
    search_word_file: str
