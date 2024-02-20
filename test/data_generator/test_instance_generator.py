from data_generator.generate_instance import generate_instance
from validation.input_validation import validate_instance


def test_generated_instance():
    res = generate_instance()
    validate_instance(res)