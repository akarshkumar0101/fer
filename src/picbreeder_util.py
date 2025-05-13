import zipfile
import xml.etree.ElementTree as ET


def _xml_to_dict(element):
    """
    Convert an XML element to a dictionary.
    """
    node = {}
    if element.attrib:
        node.update({f"@{key}": value for key, value in element.attrib.items()})
    children = list(element)
    if children:
        child_dict = {}
        for child in children:
            child_name = child.tag
            child_dict.setdefault(child_name, []).append(_xml_to_dict(child))
        for key, value in child_dict.items():
            node[key] = value if len(value) > 1 else value[0]
    else:
        if element.text and element.text.strip():
            node["#text"] = element.text.strip()
    return node

def load_zip_xml_as_dict(zip_file_path):
    """
    Load a zip file containing an XML file and return the dictionary representation of the XML.
    """
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        assert len(file_list) == 1
        for file_name in file_list:
            with zip_ref.open(file_name) as file:
                file_content = file.read().decode('utf-8')
    element = ET.fromstring(file_content)
    root = _xml_to_dict(element)
    if 'genome' not in root:
        root = dict(genome=root)
    return root
