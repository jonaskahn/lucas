"""Plugin validation utilities for Lucas.

Provide basic structure and security checks for plugin bundles. Scans files
for risky patterns and validates that required entry points exist. This is a
lightweight validator; documentation updates here do not alter runtime logic.
"""

import ast
import re
from pathlib import Path
from typing import List

from lucas.base.loggable import Loggable


class PluginValidator(Loggable):
    """Lightweight plugin validator focused on safety and structure.

    Performs quick checks on plugin directories:
    - Ensures required entry points exist (e.g., `get_metadata`, `create_agent`)
    - Scans source for potentially dangerous patterns and imports

    Some module-level imports are logged as warnings (not fatal). Dynamic
    evaluation functions (e.g., `eval`, `exec`) are treated as higher risk.
    """

    def __init__(self):
        super().__init__()
        self.dangerous_imports = {
            "os.system",
            "subprocess.call",
            "subprocess.run",
            "eval",
            "exec",
            "__import__",
            "compile",
            "open",
            "file",
            "input",
            "raw_input",
        }

        self.dangerous_patterns = [
            r"os\s*\.\s*system",
            r"subprocess\s*\.\s*(call|run|Popen)",
            r"eval\s*\(",
            r"exec\s*\(",
            r"__import__\s*\(",
            r"compile\s*\(",
        ]

    def validate_plugin_directory(self, plugin_path: Path) -> List[str]:
        """Validate a plugin directory's structure and scan its sources.

        Args:
            plugin_path: Path to the plugin directory.

        Returns:
            List[str]: Validation error messages; empty if valid.
        """
        errors = []

        try:
            plugin_file = plugin_path / "plugin.py"
            if not plugin_file.exists():
                errors.append(f"Missing required plugin.py in {plugin_path}")
                return errors

            with open(plugin_file, "r") as f:
                content = f.read()

            if "def get_metadata()" not in content:
                errors.append("Missing required function: get_metadata()")
            if "def create_agent()" not in content:
                errors.append("Missing required function: create_agent()")

            security_issues = self._scan_for_security_issues(content, plugin_file)
            if security_issues:
                errors.extend(security_issues)

            for py_file in plugin_path.glob("**/*.py"):
                if py_file.name == "__pycache__":
                    continue
                with open(py_file, "r") as f:
                    file_content = f.read()
                file_issues = self._scan_for_security_issues(file_content, py_file)
                if file_issues:
                    errors.extend(file_issues)

            return errors

        except Exception as e:
            self.logger.error(f"Validation error for {plugin_path}: {e}")
            errors.append(f"Validation exception: {str(e)}")
            return errors

    def _scan_for_security_issues(self, content: str, file_path: Path) -> List[str]:
        """Scan a source string for potentially dangerous patterns and imports.

        Args:
            content: Raw Python source code.
            file_path: File location used for reporting diagnostics.

        Returns:
            List[str]: Descriptions of detected issues.
        """
        issues = []

        # Check for dangerous patterns using regex
        for pattern in self.dangerous_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append(
                    f"Potentially dangerous pattern in {file_path}: {pattern}"
                )

        # Parse AST for more detailed analysis
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                # Check imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if self._is_dangerous_import(alias.name):
                            issues.append(
                                f"Dangerous import in {file_path}: {alias.name}"
                            )

                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        full_import = f"{module}.{alias.name}"
                        if self._is_dangerous_import(full_import):
                            issues.append(
                                f"Dangerous import in {file_path}: {full_import}"
                            )

                # Check for eval/exec calls
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ["eval", "exec", "compile", "__import__"]:
                            issues.append(
                                f"Dangerous function call in {file_path}: {node.func.id}"
                            )

        except SyntaxError as e:
            issues.append(f"Syntax error in {file_path}: {e}")

        return issues

    def _is_dangerous_import(self, import_name: str) -> bool:
        """Return True if an import path is considered dangerous.

        Args:
            import_name: Fully qualified import path (e.g., `os.system`).

        Returns:
            bool: True if deemed unsafe; otherwise False (may warn).
        """
        dangerous_modules = ["os", "subprocess", "sys", "importlib", "pickle"]

        # Check exact matches
        if import_name in self.dangerous_imports:
            return True

        # Check module-level imports
        for dangerous in dangerous_modules:
            if import_name == dangerous or import_name.startswith(f"{dangerous}."):
                self.logger.warning(f"Potentially dangerous import: {import_name}")
                return False  # For now, just warn

        return False

    def validate_plugin_code(self, code: str) -> List[str]:
        """Validate a code string for security issues using pattern scanning.

        Args:
            code: Raw Python source code.

        Returns:
            List[str]: Descriptions of detected risky patterns.
        """
        issues = []

        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append(f"Dangerous pattern found: {pattern}")

        return issues
