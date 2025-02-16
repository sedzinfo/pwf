import sublime_plugin


class ProjectVenvReplCommand(sublime_plugin.TextCommand):
    """
    Starts a SublimeREPL, attempting to use project's specified
    python interpreter.
    """

    def run(self, edit, open_file='$file'):
        """Called on project_venv_repl command"""
        cmd_list = [self.get_project_interpreter(), '-i', '-u']

        if open_file:
            cmd_list.append(open_file)

        self.repl_open(cmd_list=cmd_list)

    def get_project_interpreter(self):
        """Return the project's specified python interpreter, if any"""
        settings = self.view.settings()
        return settings.get('python_interpreter', '/usr/bin/python')

    def repl_open(self, cmd_list):
        """Open a SublimeREPL using provided commands"""
        self.view.window().run_command(
            'repl_open', {
                'encoding': 'utf8',
                'type': 'subprocess',
                'cmd': cmd_list,
                'cwd': '$file_path',
                'syntax': 'Packages/Python/Python.tmLanguage'
            }
        )
