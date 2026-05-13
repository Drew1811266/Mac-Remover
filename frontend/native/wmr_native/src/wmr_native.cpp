#include <node_api.h>

namespace {

napi_value MakeBoolean(napi_env env, bool value) {
  napi_value out;
  napi_get_boolean(env, value, &out);
  return out;
}

napi_value GetCapabilities(napi_env env, napi_callback_info /* info */) {
  napi_value result;
  napi_create_object(env, &result);

  napi_value has_core = MakeBoolean(env, true);
  napi_set_named_property(env, result, "native_core", has_core);

  napi_value cv_ready = MakeBoolean(env, false);
  napi_set_named_property(env, result, "opencv_algorithms", cv_ready);

  return result;
}

napi_value Init(napi_env env, napi_value exports) {
  napi_value fn;
  napi_create_function(env, "getCapabilities", NAPI_AUTO_LENGTH, GetCapabilities, nullptr, &fn);
  napi_set_named_property(env, exports, "getCapabilities", fn);
  return exports;
}

}  // namespace

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)
