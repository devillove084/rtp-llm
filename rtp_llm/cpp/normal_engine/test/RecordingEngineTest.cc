#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#include "rtp_llm/cpp/observability/ExecutionRecorder.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include <pybind11/embed.h>
#include "alog/Configurator.h"
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <unistd.h>

PYBIND11_EMBEDDED_MODULE(recording_engine_test_ops, m) {
    // OpDefs binds Tensor-valued defaults; initialize their Python caster first.
    pybind11::module_::import("torch");
    torch_ext::registerPyOpDefs(m);
}

namespace rtp_llm {
class RecordingEngineTest: public DeviceTestBase {
public:
    void SetUp() override {}  // Initialize CUDA only in the fresh child process.
    void runLifecycleAndSnapshot(bool tensor_capture = false);
    void runCaptureWithoutRecorder(bool enabled = true);
    void runCaptureConstructorFailure();
};

class CaptureContextTestModel: public MockModel {
public:
    CaptureContextTestModel(size_t vocab_size, std::vector<std::string>& snapshots):
        MockModel(vocab_size), snapshots_(snapshots) {}

    GptModelOutputs forward(const GptModelInputs& inputs) override {
        if (inputs.warmup || inputs.sequence_lengths.numel() != 0 || inputs.is_fake_stream
            || inputs.is_target_verify || inputs.is_draft_extend) {
            EXPECT_TRUE(inputs.tensor_capture_context_json.empty());
        } else {
            EXPECT_FALSE(inputs.tensor_capture_context_json.empty());
            snapshots_.push_back(inputs.tensor_capture_context_json);
        }
        return MockModel::forward(inputs);
    }

private:
    std::vector<std::string>& snapshots_;
};

static std::string writeCaptureConfig(const std::string& directory, bool enabled) {
    const auto path = directory + "/capture.json";
    std::ofstream file(path);
    file.exceptions(std::ios::failbit | std::ios::badbit);
    file << (enabled ? R"({"enabled":true,"authorized_capture":true,"layers":[0],"ops":["indexer.topk"],
                          "phases":["prefill"],"rank_filter":"record_owner","max_calls":1,
                          "max_bundle_bytes":1024,"max_total_bytes":1024,"max_inflight_bytes":4096})" :
                       R"({"enabled":false})");
    file.close();
    return path;
}

void RecordingEngineTest::runLifecycleAndSnapshot(bool tensor_capture) {
    if (tensor_capture && !Py_IsInitialized()) {
        py::initialize_interpreter();
    }
    DeviceTestBase::SetUp();
    char directory[] = "/tmp/rtp-recording-engine-XXXXXX";
    ASSERT_NE(mkdtemp(directory), nullptr);
    setenv("RTP_LLM_RECORD_DIR", directory, 1);
    setenv("RTP_LLM_RECORD_SESSION", "engine-test", 1);
    std::vector<std::string> snapshots;
    std::shared_ptr<NormalEngine> engine;
    if (tensor_capture) {
        ModelConfig model_config;
        RuntimeConfig runtime_config;
        KVCacheConfig kv_cache_config;
        auto params = createEngineInitParams(CustomConfig{}, model_config, runtime_config, kv_cache_config);
        params.profiling_debug_logging_config.tensor_capture_config = writeCaptureConfig(directory, true);
        NormalExecutor::test_model_factory = [&](const GptModelInitParams&) {
            return std::make_unique<CaptureContextTestModel>(model_config.vocab_size, snapshots);
        };
        engine = std::make_shared<NormalEngine>(params, nullptr);
        NormalExecutor::test_model_factory = nullptr;
    } else {
        engine = createMockEngine(CustomConfig{});
    }
    const auto batch_config =
        std::string("alog.max_msg_len=0\nalog.logger.batch_schedule=INFO, batchScheduleAppender\n")
        + "inherit.batch_schedule=false\nalog.appender.batchScheduleAppender=FileAppender\n"
        + "alog.appender.batchScheduleAppender.fileName=" + directory + "/batch_schedule.log\n"
        + "alog.appender.batchScheduleAppender.layout=PatternLayout\n"
        + "alog.appender.batchScheduleAppender.layout.LogPattern=%%m\n"
        + "alog.appender.batchScheduleAppender.async_flush=true\n";
    alog::Configurator::configureLoggerFromString(batch_config.c_str());
    auto input                             = std::make_shared<GenerateInput>();
    input->input_ids                       = torch::tensor({1, 2, 3}, torch::kInt32);
    input->generate_config                 = std::make_shared<GenerateConfig>();
    input->generate_config->max_new_tokens = 4;
    input->generate_config->is_streaming   = true;
    auto stream                            = engine->enqueue(input);
    ASSERT_TRUE(stream->nextOutput().ok());
    ASSERT_TRUE(stream->nextOutput().ok());
    ASSERT_TRUE(stream->nextOutput().ok());
    ASSERT_TRUE(stream->nextOutput().ok());
    ASSERT_FALSE(stream->nextOutput().ok());
    ASSERT_TRUE(engine->stop().ok());
    ExecutionRecorder::instance().close();
    std::ifstream file(std::string(directory) + "/batch_schedule.log");
    ASSERT_TRUE(file.good());
    const std::string batches((std::istreambuf_iterator<char>(file)), {});
    // Exercise the production snapshot path: nanosecond-sized IDs must remain
    // distinct decimal strings after a JSON round trip, even above 2^53.
    std::istringstream    lines(batches);
    std::set<std::string> execution_ids;
    for (std::string line; std::getline(lines, line);) {
        if (line.empty())
            continue;
        const auto record = autil::legacy::AnyCast<ExecutionRecorder::Json>(autil::legacy::json::ParseJson(line));
        const auto id     = autil::legacy::AnyCast<std::string>(record.at("exec_id"));
        EXPECT_GT(std::stoll(id), 9007199254740991LL);
        EXPECT_EQ(std::to_string(std::stoll(id)), id);
        EXPECT_TRUE(execution_ids.insert(id).second);
        EXPECT_EQ(autil::legacy::AnyCast<std::string>(
                      autil::legacy::json::ParseJson(ExecutionRecorder::toJson(record.at("exec_id")))),
                  id);
    }
    EXPECT_FALSE(execution_ids.empty());
    if (tensor_capture) {
        ASSERT_FALSE(snapshots.empty());
        for (const auto& snapshot : snapshots) {
            const auto context = autil::legacy::AnyCast<ExecutionRecorder::Json>(
                autil::legacy::json::ParseJson(snapshot));
            const auto id = autil::legacy::AnyCast<std::string>(context.at("exec_id"));
            EXPECT_EQ(execution_ids.count(id), 1);
            EXPECT_EQ(autil::legacy::AnyCast<std::string>(context.at("session_id")), "engine-test");
            EXPECT_EQ(autil::legacy::AnyCast<std::string>(context.at("owner_id")), ExecutionRecorder::instance().owner());
            EXPECT_EQ(autil::legacy::AnyCast<std::string>(context.at("model_role")), "target");
            const auto requests = autil::legacy::AnyCast<ExecutionRecorder::JsonArray>(context.at("requests"));
            ASSERT_EQ(requests.size(), 1);
            const auto row = autil::legacy::AnyCast<ExecutionRecorder::Json>(requests[0]);
            EXPECT_EQ(autil::legacy::AnyCast<std::string>(row.at("phase")), "prefill");
            EXPECT_EQ(ExecutionRecorder::toJson(row.at("q_len")), "3");
            EXPECT_EQ(ExecutionRecorder::toJson(row.at("kv_len")), "0");
            EXPECT_EQ(row.count("request_id"), 1);
            EXPECT_EQ(row.count("ttft_us"), 0);
        }
    }
    // Current optional fields are omitted for a single real sequence.
    for (const auto* field : {"sequence_id", "is_fake"}) {
        EXPECT_EQ(batches.find(std::string("\"") + field + "\":"), std::string::npos) << field << batches;
    }
    EXPECT_NE(batches.find("\"schema_version\":1"), std::string::npos) << batches;
    EXPECT_NE(batches.find("\"owner_id\":"), std::string::npos) << batches;
    EXPECT_NE(batches.find("\"exec_id\":"), std::string::npos) << batches;
    EXPECT_NE(batches.find("\"request_id\":"), std::string::npos) << batches;
    EXPECT_NE(batches.find("\"input_len\":3"), std::string::npos) << batches;
    EXPECT_NE(batches.find("\"reuse_len\":0"), std::string::npos) << batches;
    EXPECT_NE(batches.find("\"q_len\":3"), std::string::npos) << batches;
    EXPECT_NE(batches.find("\"kv_len\":0"), std::string::npos) << batches;
    EXPECT_NE(batches.find("\"ttft_us\":"), std::string::npos) << batches;
    EXPECT_EQ(batches.find("\"ttft_us\":null"), std::string::npos) << batches;
    const auto* decode_env    = std::getenv("RTP_LLM_RECORD_DECODE");
    const bool  record_decode = decode_env && std::string(decode_env) == "1";
    if (record_decode) {
        EXPECT_NE(batches.find("\"phase\":\"decode\""), std::string::npos) << batches;
        EXPECT_NE(batches.find("\"q_len\":1"), std::string::npos) << batches;
    } else {
        EXPECT_EQ(batches.find("\"phase\":\"decode\""), std::string::npos) << batches;
        EXPECT_EQ(batches.find("\"q_len\":1"), std::string::npos) << batches;
    }
    unsetenv("RTP_LLM_RECORD_DIR");
    unsetenv("RTP_LLM_RECORD_SESSION");
}

void RecordingEngineTest::runCaptureWithoutRecorder(bool enabled) {
    if (!Py_IsInitialized()) {
        py::initialize_interpreter();
    }
    DeviceTestBase::SetUp();
    unsetenv("RTP_LLM_RECORD_DIR");
    unsetenv("RTP_LLM_RECORD_SESSION");
    ModelConfig model_config;
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    auto params = createEngineInitParams(CustomConfig{}, model_config, runtime_config, kv_cache_config);
    char directory[] = "/tmp/rtp-capture-config-XXXXXX";
    ASSERT_NE(mkdtemp(directory), nullptr);
    params.profiling_debug_logging_config.tensor_capture_config = writeCaptureConfig(directory, enabled);
    EXPECT_EQ(PyWrappedModel::tensorCaptureConfigEnabled(params.profiling_debug_logging_config.tensor_capture_config),
              enabled);
    EXPECT_FALSE(PyWrappedModel::tensorCaptureConfigEnabled(""));
    if (enabled) {
        EXPECT_THROW(NormalExecutor(params, nullptr), std::runtime_error);
    } else {
        EXPECT_NO_THROW(NormalExecutor(params, nullptr));
    }
}

void RecordingEngineTest::runCaptureConstructorFailure() {
    if (!Py_IsInitialized()) {
        py::initialize_interpreter();
    }
    DeviceTestBase::SetUp();
    py::gil_scoped_acquire gil;
    // Use the production registrations before C++ passes init_resources to Python.
    auto test_ops = py::module_::import("recording_engine_test_ops");
    auto module = py::module_::import("rtp_llm.utils.tensor_capture");
    auto original_recorder = module.attr("TensorCaptureRecorder").cast<py::object>();
    py::dict scope;
    scope["PyModelInitResources"] = test_ops.attr("PyModelInitResources");
    py::exec(R"(
import ctypes
events = []
fail_close = False
class Recorder:
    def __init__(self, config_path, *, tp_rank, world_rank):
        events.append('created')
    def close(self, timeout=5.0):
        assert ctypes.pythonapi.PyGILState_Check() == 1
        assert timeout == 5.0
        events.append('closed')
        if fail_close:
            raise RuntimeError('close-failure')
        return True
    def __del__(self):
        events.append('destroyed')
class Model:
    def initialize(self, resources):
        assert isinstance(resources, PyModelInitResources)
        events.append('initialized')
        raise RuntimeError('expected-model-init-failure')
model = Model()
)", scope);
    module.attr("TensorCaptureRecorder") = scope["Recorder"];
    try {
        ModelConfig model_config;
        RuntimeConfig runtime_config;
        KVCacheConfig kv_cache_config;
        auto params = createEngineInitParams(CustomConfig{}, model_config, runtime_config, kv_cache_config);
        char directory[] = "/tmp/rtp-capture-constructor-XXXXXX";
        if (!mkdtemp(directory)) {
            throw std::runtime_error("failed to create constructor test directory");
        }
        params.profiling_debug_logging_config.tensor_capture_config = writeCaptureConfig(directory, false);
        params.py_model = scope["model"].cast<py::object>();
        for (bool fail_close : {false, true}) {
            scope["fail_close"] = fail_close;
            scope["events"].attr("clear")();
            try {
                py::gil_scoped_release release;
                NormalExecutor executor(params, nullptr, /*warm_up=*/true);
                ADD_FAILURE() << "model initialization should fail";
            } catch (const py::error_already_set& e) {
                EXPECT_NE(std::string(e.what()).find("expected-model-init-failure"), std::string::npos);
            }
            EXPECT_EQ(scope["events"].cast<std::vector<std::string>>(),
                      (std::vector<std::string>{"created", "initialized", "closed", "destroyed"}));
        }
    } catch (...) {
        module.attr("TensorCaptureRecorder") = original_recorder;
        throw;
    }
    module.attr("TensorCaptureRecorder") = original_recorder;
}

TEST_F(RecordingEngineTest, CaptureConstructorFailureClosesUnderGil) {
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    ASSERT_EXIT(
        {
            runCaptureConstructorFailure();
            _exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

TEST_F(RecordingEngineTest, DisabledCaptureDoesNotRequireRecorder) {
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    ASSERT_EXIT(
        {
            runCaptureWithoutRecorder(false);
            _exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

TEST_F(RecordingEngineTest, CaptureRequiresReadyRecorder) {
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    ASSERT_EXIT(
        {
            runCaptureWithoutRecorder();
            _exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

TEST_F(RecordingEngineTest, CaptureContextUsesRecordedBatch) {
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    ASSERT_EXIT(
        {
            // Decode batch recording must not enable decode tensor capture.
            setenv("RTP_LLM_RECORD_DECODE", "1", 1);
            runLifecycleAndSnapshot(true);
            _exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

TEST_F(RecordingEngineTest, LifecycleAndSnapshot) {
    // Re-exec rather than fork with an initialized CUDA runtime. Each repeat
    // gets fresh recorder and alog state; failures in the child reach the parent.
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    ASSERT_EXIT(
        {
            runLifecycleAndSnapshot();
            _exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

}  // namespace rtp_llm
