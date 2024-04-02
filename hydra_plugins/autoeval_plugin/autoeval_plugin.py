from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin

from urartu.utils.user import get_current_user

current_user = get_current_user()


class urartuPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.prepend(provider="urartu", path=f"pkg://configs")
        search_path.prepend(provider="urartu", path=f"file://configs")
        search_path.prepend(provider="urartu", path=f"pkg://configs_{current_user}")
        search_path.prepend(provider="urartu", path=f"file://configs_{current_user}")
