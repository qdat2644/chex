import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from app.experiment_integrity import canonical_json_sha256, compute_file_sha256, restore_safe_rng_state
from scripts.calibrate import validate_calibration_config
from scripts.train import checkpoint_payload, load_resume_checkpoint, restore_training_state, run_epoch, set_seed
from scripts.package_kaggle_artifact import package_artifact, verify_artifact_zip
from scripts.verify_resume_bundle import verify_resume_bundle
from tests import test_hardening_production as fixtures


class RemainingBlockers(unittest.TestCase):
    def test_all_training_seeds_with_shared_split(self):
        for seed in range(42, 47):
            with self.subTest(seed=seed), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run, cal, manifest, config = fixtures.TestProductionHardening()._artifact_fixture(root)
                resolved_path = run / 'resolved_config.json'
                resolved = json.loads(resolved_path.read_text())
                resolved['seed'] = seed
                resolved_path.write_text(json.dumps(resolved))
                digest = canonical_json_sha256(resolved)
                for name in ('best.pt', 'last.pt'):
                    ck = torch.load(run / name, weights_only=True)
                    ck.update(seed=seed, resolved_config_sha256=digest)
                    torch.save(ck, run / name)
                for path in (run / 'run_manifest.json', cal):
                    data = json.loads(path.read_text())
                    data.update(seed=seed, resolved_config_sha256=digest)
                    if path == cal:
                        data['checkpoint_sha256'] = compute_file_sha256(run / 'best.pt')
                    else:
                        data['best_checkpoint_sha256'] = compute_file_sha256(run / 'best.pt')
                        data['last_checkpoint_sha256'] = compute_file_sha256(run / 'last.pt')
                    path.write_text(json.dumps(data))
                out = package_artifact(run, cal, manifest, config, root / 'out.zip')
                ledger = verify_artifact_zip(out)
                self.assertEqual(ledger['training_seed'], seed)
                self.assertEqual(ledger['split_seed'], 42)

    def test_calibration_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'protocol.yaml'
            config.write_text('protocol_version: 0.1')
            resolved = root / 'resolved_config.json'
            resolved.write_text('{}')
            ck = dict(architecture='tiny', seed=43, labels=['L'], run_mode='full',
                      config_sha256=compute_file_sha256(config), resolved_config_sha256=canonical_json_sha256({}),
                      manifest_sha256='abc', git_commit='abc')
            validate_calibration_config(ck, root / 'best.pt', config)
            with self.assertRaisesRegex(RuntimeError, 'run-mode'):
                validate_calibration_config(ck, root / 'best.pt', config, 'smoke')
            with self.assertRaisesRegex(RuntimeError, 'limit'):
                validate_calibration_config(ck, root / 'best.pt', config, limit=0)
            resolved.unlink()
            with self.assertRaisesRegex(RuntimeError, 'requires'):
                validate_calibration_config(ck, root / 'best.pt', config)
            resolved.write_text('{"changed":true}')
            with self.assertRaisesRegex(RuntimeError, 'hash mismatch'):
                validate_calibration_config(ck, root / 'best.pt', config)
            resolved.write_text('{}')
            config.write_text('changed')
            with self.assertRaisesRegex(RuntimeError, 'YAML hash'):
                validate_calibration_config(ck, root / 'best.pt', config)

    def test_bundle_external_digest_before_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bundle.zip'
            path.write_bytes(b'not a zip')
            with patch('scripts.verify_resume_bundle.zipfile.ZipFile') as opener:
                for expected in (None, '0' * 64):
                    with self.assertRaises(RuntimeError):
                        verify_resume_bundle(path, expected_sha256=expected)
                opener.assert_not_called()

    def test_missing_resume_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'last.pt'
            for field in ('best_epoch', 'patience_counter'):
                ck = dict(best_epoch=1, patience_counter=2)
                del ck[field]
                torch.save(ck, path)
                with self.assertRaisesRegex(RuntimeError, field):
                    load_resume_checkpoint(path, compute_file_sha256(path))

    def test_cuda_rng_fail_closed(self):
        with patch('torch.cuda.device_count', return_value=2):
            with self.assertRaisesRegex(RuntimeError, 'count'):
                restore_safe_rng_state({'torch_cuda': [torch.zeros(1)]}, strict=True)
        with patch('torch.cuda.device_count', return_value=1), patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.set_rng_state_all', side_effect=ValueError('bad')):
            with self.assertRaisesRegex(RuntimeError, 'restoration failed'):
                restore_safe_rng_state({'torch_cuda': [torch.zeros(1)]}, strict=True)

    def test_production_resume_equivalence(self):
        x = torch.arange(24, dtype=torch.float32).reshape(8, 3) / 24
        y = (x[:, :1] > .4).float()
        dataset = TensorDataset(x, y, torch.ones_like(y))

        def fresh(seed):
            set_seed(seed)
            model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.2), torch.nn.Linear(4, 1))
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
            generator = torch.Generator().manual_seed(seed)
            return model, optimizer, scheduler, generator

        def epoch(state, number, history, best_epoch, patience):
            model, optimizer, scheduler, generator = state
            loader = DataLoader(dataset, batch_size=2, shuffle=True, generator=generator)
            loss, _, _, _ = run_epoch(model, loader, torch.nn.BCEWithLogitsLoss(reduction='none'), torch.device('cpu'), optimizer)
            scheduler.step()
            history.append({'epoch': number, 'loss': loss})
            return checkpoint_payload(model, optimizer, scheduler, None, number, 2, .8, 'tiny', 42,
                'config', 'resolved', 'manifest', ['L'], 'smoke', generator=generator,
                history=history, best_epoch=best_epoch, patience_counter=patience)

        continuous = fresh(42)
        history = []
        epoch(continuous, 1, history, 1, 3)
        expected = copy.deepcopy(epoch(continuous, 2, history, 1, 4))
        continuous[0].eval()
        prediction = continuous[0](x).detach()
        first = fresh(42)
        checkpoint = epoch(first, 1, [], 1, 3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'last.pt'
            torch.save(checkpoint, path)
            resumed = fresh(999)
            loaded, _ = load_resume_checkpoint(path, compute_file_sha256(path))
            next_epoch, best, best_epoch, patience, history = restore_training_state(loaded, *resumed[:3], generator=resumed[3])
            self.assertEqual((next_epoch, best, best_epoch, patience), (2, .8, 1, 3))
            actual = epoch(resumed, next_epoch, history, best_epoch, patience + 1)
            def compare(a, b):
                if isinstance(a, torch.Tensor):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                elif isinstance(a, dict):
                    self.assertEqual(a.keys(), b.keys())
                    for key in a: compare(a[key], b[key])
                elif isinstance(a, (list, tuple)):
                    self.assertEqual(len(a), len(b))
                    for left, right in zip(a, b): compare(left, right)
                else: self.assertEqual(a, b)
            for field in ('model_state', 'optimizer_state', 'scheduler_state', 'epoch', 'best_val_auc', 'best_epoch', 'patience_counter'):
                compare(expected[field], actual[field])
            compare(expected['metadata']['metrics_history'], actual['metadata']['metrics_history'])
            resumed[0].eval()
            torch.testing.assert_close(prediction, resumed[0](x), rtol=0, atol=0)
