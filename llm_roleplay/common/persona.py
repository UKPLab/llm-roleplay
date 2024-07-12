import hashlib
import random
from typing import Dict, List, Tuple


class Persona:
    @staticmethod
    def get_personas(cfg) -> List[Tuple[str, Dict[str, str]]]:
        if "fixed" in cfg:
            personas: List[Tuple[str, Dict[str, str]]] = []
            for person in cfg.fixed:
                persona = cfg.prompt
                features = person["person"]

                for feature_name in features.keys():
                    persona = persona.replace(
                        f"<{feature_name.upper()}>", features[feature_name]
                    )

                persona_hash = hashlib.md5(str(features).encode()).hexdigest()
                personas.append((persona, persona_hash))
            return personas
        else:
            return Persona.generate_personas(cfg)

    @staticmethod
    def generate_personas(cfg) -> List[Tuple[str, Dict[str, str]]]:
        personas: List[Tuple[str, Dict[str, str]]] = []
        for _ in range(cfg.num_personas):
            persona = cfg.prompt
            chosen_features = {}
            for feature_name in cfg.features.keys():
                feature = random.choice(cfg.features[feature_name])
                persona = persona.replace(f"<{feature_name.upper()}>", feature)
                chosen_features[feature_name] = feature
            persona_hash = hashlib.md5(str(chosen_features).encode()).hexdigest()
            personas.append((persona, persona_hash))
        return personas
