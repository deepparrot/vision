#!/usr/bin/env python3

import jinja2, yaml
import os.path

def workflow(btype, os, python_version, cu_version, unicode, prefix='', upload=False):
    w = []
    unicode_suffix = "u" if unicode else ""
    workflow_name = f"{prefix}binary_{os}_{btype}_py{python_version}{unicode_suffix}_{cu_version}"
    d = {
        "name": workflow_name,
        "python_version": python_version,
        "cu_version": cu_version,
    }
    if unicode:
        d["unicode_abi"] = "1"
    if cu_version == "cu92":
        d["wheel_docker_image"] = "soumith/manylinux-cuda92"
    w.append({f"binary_{os}_{btype}": d})

    if upload:
        d_upload = {
            "name": f"{workflow_name}_upload",
            "context": "org-member",
            "requires": [workflow_name],
        }
        if btype == 'wheel':
            d_upload["subfolder"] = "" if os == 'macos' else cu_version + "/"
        w.append({f"binary_{btype}_upload": d_upload})

    return w

def workflows(prefix='', upload=False, indentation=6):
    w = []
    for btype in ["wheel", "conda"]:
        for os in ["linux", "macos"]:
            for python_version in ["2.7", "3.5", "3.6", "3.7"]:
                for cu_version in (["cpu", "cu92", "cu100"] if os == "linux" else ["cpu"]):
                    for unicode in ([False, True] if btype == "wheel" and python_version == "2.7" else [False]):
                        w += workflow(btype, os, python_version, cu_version, unicode, prefix=prefix, upload=upload)
    return ("\n" + " "*indentation).join(yaml.dump(w).splitlines())

d = os.path.dirname(__file__)
env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(d),
    lstrip_blocks=True,
    autoescape=False,
)
with open(os.path.join(d, 'config.yml'), 'w') as f:
    f.write(env.get_template('config.yml.in').render(workflows=workflows))

