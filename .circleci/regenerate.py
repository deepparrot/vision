#!/usr/bin/env python3

import jinja2, yaml
import os.path


class Workflow:
    def __init__(self, name, members):
        self.name = name
        self.members = members

    def addIf(self, condition, items):
        if condition:
            self.members.update(items)

    def includeIn(self, workflow_list):
        workflow_list.append({self.name: self.members})


def workflow(btype, os, python_version, cu_version, unicode, prefix='', upload=False):

    workflow_list = []

    unicode_suffix = "u" if unicode else ""
    base_workflow_name = f"{prefix}binary_{os}_{btype}_py{python_version}{unicode_suffix}_{cu_version}"

    w = Workflow(f"binary_{os}_{btype}", {
        "name": base_workflow_name,
        "python_version": python_version,
        "cu_version": cu_version,
    })

    w.addIf(unicode, {
        "unicode_abi": "1"
    })

    w.addIf(cu_version == "cu92", {
        "wheel_docker_image": "soumith/manylinux-cuda92"
    })

    w.includeIn(workflow_list)

    if upload:
        w2 = Workflow(f"binary_{btype}_upload", {
            "name": f"{base_workflow_name}_upload",
            "context": "org-member",
            "requires": [base_workflow_name],
        })

        w2.addIf(btype == "wheel", {
            "subfolder": "" if os == 'macos' else cu_version + "/"
        })

        w2.includeIn(workflow_list)

    return workflow_list

def workflows(prefix='', upload=False, indentation=6):
    w = []
    for btype in ["wheel", "conda"]:
        for os in ["linux", "macos"]:
            for python_version in ["2.7", "3.5", "3.6", "3.7"]:
                for cu_version in (["cpu", "cu92", "cu100"] if os == "linux" else ["cpu"]):
                    for unicode in ([False, True] if btype == "wheel" and python_version == "2.7" else [False]):
                        w += workflow(btype, os, python_version, cu_version, unicode, prefix=prefix, upload=upload)
    return ("\n" + " " * indentation).join(yaml.dump(w).splitlines())

d = os.path.dirname(__file__)
env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(d),
    lstrip_blocks=True,
    autoescape=False,
)
with open(os.path.join(d, 'config.yml'), 'w') as f:
    f.write(env.get_template('config.yml.in').render(workflows=workflows))

