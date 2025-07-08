from typing import Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


class CloudSettings(BaseSettings):
    client_id: str = Field(..., alias="TYPEDEF_CLIENT_ID")
    client_secret: str = Field(..., alias="TYPEDEF_CLIENT_SECRET")
    auth_provider_uri: str = Field(..., alias="CLOUD_SESSION_AUTH_PROVIDER_URI")
    typedef_instance: str = Field(
        default="dev1", alias="CLOUD_SESSION_TYPEDEF_INSTANCE"
    )
    typedef_environment: str = Field(
        default="dev", alias="CLOUD_SESSION_TYPEDEF_ENVIRONMENT"
    )
    # These endpoints get constructed from the instance, if not set in env vars
    hasura_graphql_uri: Optional[str] = None
    hasura_graphql_ws_uri: Optional[str] = None
    api_auth_uri: Optional[str] = None
    entrypoint_uri: Optional[str] = None
    # user token can optionally be initialized in env var
    client_token: Optional[str] = None
    test_engine_grpc_url: Optional[str] = None
    test_engine_arrow_url: Optional[str] = None

    class Config:
        env_prefix = "CLOUD_SESSION_"

    @model_validator(mode="after")
    def construct_endpoints(self):
        # construct the endpoints after env vars are set
        if not self.hasura_graphql_uri:
            self.hasura_graphql_uri = (
                f"https://api.{self.typedef_instance}.typedef.engineering/v1/graphql"
            )
        if not self.hasura_graphql_ws_uri:
            self.hasura_graphql_ws_uri = (
                f"wss://api.{self.typedef_instance}.typedef.engineering/v1/graphql"
            )
        if not self.api_auth_uri:
            self.api_auth_uri = f"https://api.{self.typedef_instance}.typedef.engineering/v1/auth/token/authorize"
        if not self.entrypoint_uri:
            self.entrypoint_uri = (
                f"entrypoint.{self.typedef_instance}.typedef.engineering"
            )

    def __str__(self):
        return (
            f"  settings.client_id={self.client_id}\n"
            f"  settings.client_secret={'set' if self.client_secret else 'not set'}\n"
            f"  settings.auth_provider_uri={self.auth_provider_uri}\n"
            f"  settings.hasura_graphql_uri={self.hasura_graphql_uri}\n"
            f"  settings.hasura_graphql_ws_uri={self.hasura_graphql_ws_uri}\n"
            f"  settings.api_auth_uri={self.api_auth_uri}\n"
            f"  settings.entrypoint_uri={self.entrypoint_uri}\n"
        )
