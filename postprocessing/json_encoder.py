import json


class MyJSONEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        super(MyJSONEncoder, self).__init__(*args, **kwargs)
        self.current_indent = 0
        self.current_indent_str = ""

    def encode(self, obj):
        if isinstance(obj, (list, tuple)):
            primitives_only = True
            for item in obj:
                if isinstance(item, (list, tuple, dict)):
                    primitives_only = False
                    break
            output = []
            if primitives_only:
                for item in obj:
                    output.append(json.dumps(item))
                return "[" + ", ".join(output) + "]"
            else:
                self.current_indent += self.indent
                self.current_indent_str = "".join([" " for x in range(self.current_indent)])
                for item in obj:
                    output.append(self.current_indent_str + self.encode(item))
                self.current_indent -= self.indent
                self.current_indent_str = "".join([" " for x in range(self.current_indent)])
                return "[\n" + ",\n".join(output) + "\n" + self.current_indent_str + "]"
        elif isinstance(obj, dict):
            output = []
            self.current_indent += self.indent
            self.current_indent_str = "".join([" " for x in range(self.current_indent)])
            for key, value in obj.items():
                output.append(self.current_indent_str + json.dumps(key) + ": " + self.encode(value))
            self.current_indent -= self.indent
            self.current_indent_str = "".join([" " for x in range(self.current_indent)])
            return "{\n" + ",\n".join(output) + "\n" + self.current_indent_str + "}"
        else:
            return json.dumps(obj)
