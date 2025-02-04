from typing import Any, Dict, Optional
from hub.constants import AGREEMENT_FILENAME, HUB_CLOUD_DEV_USERNAME
from hub.core.dataset import Dataset
from hub.client.client import HubBackendClient
from hub.client.log import logger
from hub.util.agreement import handle_dataset_agreement
from hub.util.path import is_hub_cloud_path
from warnings import warn


class HubCloudDataset(Dataset):
    def first_load_init(self):
        if self.is_first_load:
            self._set_org_and_name()
            if self.is_actually_cloud:
                handle_dataset_agreement(
                    self.agreement, self.path, self.ds_name, self.org_id
                )
                logger.info(
                    f"This dataset can be visualized at https://app.activeloop.ai/{self.org_id}/{self.ds_name}."
                )
            else:
                # NOTE: this can happen if you override `hub.core.dataset.FORCE_CLASS`
                warn(
                    f'Created a hub cloud dataset @ "{self.path}" which does not have the "hub://" prefix. Note: this dataset should only be used for testing!'
                )

    @property
    def client(self):
        if self._client is None:
            self.__dict__["_client"] = HubBackendClient(token=self._token)
        return self._client

    @property
    def is_actually_cloud(self) -> bool:
        """Datasets that are connected to hub cloud can still technically be stored anywhere.
        If a dataset is hub cloud but stored without `hub://` prefix, it should only be used for testing.
        """
        return is_hub_cloud_path(self.path)  # type: ignore

    @property
    def token(self):
        """Get attached token of the dataset"""
        if self._token is None:
            self.__dict__["_token"] = self.client.get_token()
        return self._token

    def _set_org_and_name(self):
        if self.is_actually_cloud:
            if self.org_id is not None:
                return
            split_path = self.path.split("/")
            org_id, ds_name = split_path[2], split_path[3]
        else:
            # if this dataset isn't actually pointing to a datset in the cloud
            # a.k.a this dataset is trying to simulate a hub cloud dataset
            # it's safe to assume they want to use the dev org
            org_id = HUB_CLOUD_DEV_USERNAME
            ds_name = self.path.replace("/", "_").replace(".", "")
        self.__dict__["org_id"] = org_id
        self.__dict__["ds_name"] = ds_name

    def _register_dataset(self):
        # called in super()._populate_meta
        self._set_org_and_name()
        self.client.create_dataset_entry(
            self.org_id,
            self.ds_name,
            self.version_state["meta"].__getstate__(),
            public=self.public,
        )

    def make_public(self):
        self._set_org_and_name()
        if not self.public:
            self.client.update_privacy(self.org_id, self.ds_name, public=True)
            self.__dict__["public"] = True

    def make_private(self):
        self._set_org_and_name()
        if self.public:
            self.client.update_privacy(self.org_id, self.ds_name, public=False)
            self.__dict__["public"] = False

    def delete(self, large_ok=False):
        super().delete(large_ok=large_ok)

        self.client.delete_dataset_entry(self.org_id, self.ds_name)

    @property
    def agreement(self) -> Optional[str]:
        try:
            agreement_bytes = self.storage[AGREEMENT_FILENAME]  # type: ignore
            return agreement_bytes.decode("utf-8")
        except KeyError:
            return None

    def add_agreeement(self, agreement: str):
        self.storage.check_readonly()  # type: ignore
        self.storage[AGREEMENT_FILENAME] = agreement.encode("utf-8")  # type: ignore

    def __getstate__(self) -> Dict[str, Any]:
        self._set_org_and_name()
        state = super().__getstate__()
        return state

    def __setstate__(self, state: Dict[str, Any]):
        super().__setstate__(state)
        self._client = None
        self.first_load_init()
